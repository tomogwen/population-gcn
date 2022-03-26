from numpy import promote_types
from gcn.layers import *
from gcn.metrics import *
import tensorflow

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'regularizer'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        regularize = kwargs.get('regularizer', False)
        self.regularize = regularize

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.auc = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

        # Regularuzation
        if self.regularize:
            self.loss += self._regularization_term(self.outputs)
    
    def _regularization_term_old(self, outputs):
        #return self.placeholders['reg']*self.loss
        pred = tensorflow.nn.softmax(outputs) #tensorflow.Variable(self.outputs[0], dtype=tf.float32, validate_shape=False).set_shape([2,])
        # Regularization term (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6137441)
        outputs_0 = tensorflow.einsum('n,nm->nm', 
          self.placeholders['sex_mask_0'], outputs)
        outputs_1 = tensorflow.einsum('n,nm->nm', 
          self.placeholders['sex_mask_1'], outputs)
        pred_mean_0 = tensorflow.math.reduce_mean(tensorflow.nn.softmax(outputs_0), 0)[0]  
        #tensorflow.nn.softmax(tensorflow.math.reduce_mean(outputs_0, 0))[0]
        pred_mean_1 = tensorflow.math.reduce_mean(tensorflow.nn.softmax(outputs_1), 0)[0] 
        #tensorflow.nn.softmax(tensorflow.math.reduce_mean(outputs_1, 0))[0]
        
        pred_0 = tensorflow.einsum('n,nm->nm', 
          self.placeholders['sex_mask_0'], pred) # pred * self.placeholders['sex_mask_0']
        pred_1 = tensorflow.einsum('n,nm->nm', 
          self.placeholders['sex_mask_1'], pred)
        
        s_mean_0 = tensorflow.math.reduce_mean(self.placeholders['sex_mask_0'])
        s_mean_1 = tensorflow.math.reduce_mean(self.placeholders['sex_mask_1'])
        p_y = s_mean_0 * pred_mean_0 + s_mean_1 * pred_mean_1

        loss = tensorflow.math.reduce_sum(pred_0[:,0] * tensorflow.math.log(pred_mean_0 / p_y))
        loss += tensorflow.math.reduce_sum(pred_1[:,0] * tensorflow.math.log(pred_mean_1 / p_y))

        loss += tensorflow.math.reduce_sum(pred_0[:,1] * tensorflow.math.log((1. - pred_mean_0) / (1. - p_y)))
        loss += tensorflow.math.reduce_sum(pred_1[:,1] * tensorflow.math.log((1. - pred_mean_1) / (1. - p_y)))

        return self.placeholders['reg']*loss

    def _regularization_term(self, outputs):
        # For the mutual information, 
        # Pr[y|s] = sum{(xi,si),si=s} sigma(xi,si) / #D[xs]
        outputs = tensorflow.math.sigmoid(outputs)
        output_m = tensorflow.einsum('n,nm->nm', 
          self.placeholders['sex_mask_0'], outputs)
        output_f = tensorflow.einsum('n,nm->nm', 
          self.placeholders['sex_mask_1'], outputs)
        #D[xs]
        N_male = tensorflow.math.reduce_sum(self.placeholders['sex_mask_0'])
        N_female = tensorflow.math.reduce_sum(self.placeholders['sex_mask_1'])
        Dxisi = tensorflow.stack((N_male,N_female),axis=0)
        # Pr[y|s]
        y_pred_female = tensorflow.math.reduce_sum(output_f)
        y_pred_male   = tensorflow.math.reduce_sum(output_m)
        P_ys = tensorflow.stack((y_pred_male,y_pred_female),axis=0) / Dxisi
        # Pr[y]
        P = tensorflow.concat((output_f,output_m), 0)
        #P_y = tensorflow.math.reduce_sum(P) / (x_female.shape[0]+x_male.shape[0])
        P_y = tensorflow.math.reduce_sum(P) / (N_female+N_male)
        # P(siyi)
        P_s1y1 = tensorflow.math.log(P_ys[1]) - tensorflow.math.log(P_y)
        P_s1y0 = tensorflow.math.log(1-P_ys[1]) - tensorflow.math.log(1-P_y)
        P_s0y1 = tensorflow.math.log(P_ys[0]) - tensorflow.math.log(P_y)
        P_s0y0 = tensorflow.math.log(1-P_ys[0]) - tensorflow.math.log(1-P_y)
        # PI
        PI_s1y1 = output_f * P_s1y1
        PI_s1y0 =(1- output_f) * P_s1y0
        PI_s0y1 = output_m * P_s0y1
        PI_s0y0 = (1- output_m )* P_s0y0
        PI = tensorflow.math.reduce_sum(PI_s1y1) + tensorflow.math.reduce_sum(PI_s1y0) + tensorflow.math.reduce_sum(PI_s0y1) + tensorflow.math.reduce_sum(PI_s0y0)
        PI = self.placeholders['reg'] * PI
        return PI

# class PRLoss():#using linear
#      def __init__(self, eta=1.0):
#         super(PRLoss, self).__init__()
#         self.eta = eta
#      def forward(self,output_f,output_m):
#         # For the mutual information, 
#         # Pr[y|s] = sum{(xi,si),si=s} sigma(xi,si) / #D[xs]
#         #D[xs]
#         N_female = t.tensor(output_f.shape[0])
#         N_male   = t.tensor(output_m.shape[0])
#         Dxisi = t.stack((N_male,N_female),axis=0) #male sample, #female sample
#         # Pr[y|s]
#         y_pred_female = t.sum(output_f)
#         y_pred_male   = t.sum(output_m)
#         P_ys = t.stack((y_pred_male,y_pred_female),axis=0) / Dxisi
#         # Pr[y]
#         P = t.cat((output_f,output_m),0)
#         P_y = t.sum(P) / (x_female.shape[0]+x_male.shape[0])
#         # P(siyi)
#         P_s1y1 = t.log(P_ys[1]) - t.log(P_y)
#         P_s1y0 = t.log(1-P_ys[1]) - t.log(1-P_y)
#         P_s0y1 = t.log(P_ys[0]) - t.log(P_y)
#         P_s0y0 = t.log(1-P_ys[0]) - t.log(1-P_y)
#         # PI
#         PI_s1y1 = output_f * P_s1y1
#         PI_s1y0 =(1- output_f) * P_s1y0
#         PI_s0y1 = output_m * P_s0y1
#         PI_s0y0 = (1- output_m )* P_s0y0
#         PI = t.sum(PI_s1y1) + t.sum(PI_s1y0) + t.sum(PI_s0y1) + t.sum(PI_s0y0)
#         PI = self.eta * PI
#         return PI
        

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.auc = masked_auc(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
                
        
class Deep_GCN(GCN):
    
    def __init__(self, placeholders, input_dim, depth, **kwargs):
        self.depth = depth
        super(Deep_GCN, self).__init__(placeholders, input_dim, **kwargs)
        
    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        for i in range(self.depth):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden1,
                                               placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
