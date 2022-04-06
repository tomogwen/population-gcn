# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira
# Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import time
import argparse

import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
import scipy.io as sio
import os

import ABIDEParser as Reader
import train_GCN as Train


# Prepares the training/test data for each cross validation fold and trains
# the GCN
def train_fold_sim(train_ind, val_ind, test_ind, graph_feat, features, y, y_data,
               params,
               sex_data=None, stratify=False, fold_index=None, baseline=False,
               transfer_learning=False, model_number=None, model_location=None,
               group=None, random_seed=False
               ):
    """
        train_ind       : indices of the training samples
        test_ind        : indices of the test samples
        val_ind         : indices of the validation samples
        graph_feat      : population graph computed from phenotypic measures
        num_subjects x num_subjects
        features        : feature vectors num_subjects x num_features
        y               : ground truth labels (num_subjects x 1)
        y_data          : ground truth labels - different representation (
        num_subjects x 2)
        params          : dictionnary of GCNs parameters
        subject_IDs     : list of subject IDs

    returns:

        test_acc    : average accuracy over the test samples using GCNs
        test_auc    : average area under curve over the test samples using GCNs
        lin_acc     : average accuracy over the test samples using the
        linear classifier
        lin_auc     : average area under curve over the test samples using
        the linear classifier
        fold_size   : number of test samples
    """

    print(len(train_ind))

    # selection of a subset of data if running experiments with a subset of
    # the training set
    # labeled_ind = Reader.site_percentage(train_ind, params['num_training'],
    #                                     subject_IDs)

    # feature selection/dimensionality reduction step
    x_data = Reader.feature_selection(features, y, labeled_ind,
                                      params['num_features'])

    fold_size = len(test_ind)

    # Calculate all pairwise distances
    distv = distance.pdist(x_data, metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph = graph_feat * sparse_graph

    # Linear classifier
    clf = RidgeClassifier()
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # Compute the accuracy
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # Compute the AUC
    pred = clf.decision_function(x_data[test_ind, :])
    lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)

    print("Linear Accuracy: " + str(lin_acc))

    # For Baseline results
    pred = clf.decision_function(x_data)
    temp = np.zeros((2, len(pred)))
    temp[1][pred > 0] = 1
    temp[0][temp[1] != 1] = 0
    pred = temp.T

    pred_train = clf.decision_function(x_data[train_ind, :])
    temp = np.zeros((2, len(pred_train)))
    temp[1][pred_train > 0] = 1
    temp[0][temp[1] != 1] = 0
    pred_train = temp.T

    test_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    test_acc = int(round(test_acc * len(test_ind)))
    test_auc = lin_auc

    if baseline == False:
        # Classification with GCNs
        if transfer_learning == True:
            print('test')
            pred, test_acc, test_auc, pred_train = Train.run_training_transfer(
                final_graph,
                sparse.coo_matrix(
                    x_data).tolil(), y_data,
                train_ind, val_ind,
                test_ind, params, sex_data,
                stratify, fold_index, model_location,
                model_number, group, random_seed)
        else:
            pred, test_acc, test_auc, pred_train = Train.run_training(
                final_graph,
                sparse.coo_matrix(
                    x_data).tolil(), y_data,
                train_ind, val_ind,
                test_ind, params, sex_data,
                stratify, fold_index, model_number,
                random_seed)

        print(test_acc)

        # return number of correctly classified samples instead of percentage
        test_acc = int(round(test_acc * len(test_ind)))
        lin_acc = int(round(lin_acc * len(test_ind)))

    return pred, test_acc, test_auc, lin_acc, lin_auc, fold_size, pred_train


# For compatibility with Pool.map
def train_fold_thread_sim(
        indices_tuple, fold_index=None, *, graph_feat, features, y, y_data,
        params, subject_IDs,
        sex_data=None, stratify=False, baseline=False, transfer_learning=False,
        model_number=None, model_location=None, group=None, random_seed=False
):
    """
        indices tuple   : tuple of indices of the training, test,
        and validation samples
        graph_feat      : population graph computed from phenotypic measures
        num_subjects x num_subjects
        features        : feature vectors num_subjects x num_features
        y               : ground truth labels (num_subjects x 1)
        y_data          : ground truth labels - different representation (
        num_subjects x 2)
        params          : dictionary of GCNs parameters
        subject_IDs     : list of subject IDs
    returns:
        test_acc    : average accuracy over the test samples using GCNs
        test_auc    : average area under curve over the test samples using GCNs
        lin_acc     : average accuracy over the test samples using the
        linear classifier
        lin_auc     : average area under curve over the test samples using
        the linear classifier
        fold_size   : number of test samples
        test_ind    : indices of the test samples (for keeping track)
    """
    train_ind, val_ind, test_ind = indices_tuple
    pred, test_acc, test_auc, lin_acc, lin_auc, fold_size, pred_train = \
        train_fold_sim(
        train_ind,
        val_ind,
        test_ind,
        graph_feat,
        features,
        y,
        y_data,
        params,
        sex_data,
        stratify,
        fold_index,
        baseline,
        transfer_learning,
        model_number,
        model_location,
        group,
        random_seed
    )
    return pred, test_acc, test_auc, lin_acc, lin_auc, fold_size, test_ind, \
           pred_train

