import tensorflow as tf 
import numpy as np 
import os
from rbn import RestrictedBoltzmanMachinesLayer 
from tqdm import tqdm
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


class DeepBeliefNetwork:
    #include all hyperparameters in init like learning rate, batch size, decay_rate .....
    def __init__(self, pretrain_iterations,rbm_layers,rbm_activations,freeze_rbms,
                 dense_layers,dense_activations,batch_normalization,output_activation,
                 batch_size,learning_rate,beta1,keep_chance=0.5, 
                 name="dbn" ):
        #RBM Network
        self.rbms = []
        for i, layer in enumerate(rbm_layers):
            rbm_layer_name = "rbm_"+str(layer) + str(i)
            self.rbms.append(RestrictedBoltzmanMachinesLayer(layer, rbm_layer_name, freeze_rbms[i]))
        self.pretrain_iterations = pretrain_iterations
        self.learning_rbm_rate = learning_rate
        


        #Dense Network
        self.dense_layers = dense_layers
        self.dense_activation = dense_activations
        self.output_activation = output_activation
        self.batch_normalization = batch_normalization
        self.keep_chance = keep_chance
        self.name = name

        #Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.learning_rate_decay_examples = 4000
        self.learning_rate_decay = 0.96

        self.graph = None
        
    
        
    
    def pretrain(self,features):
        for i, layer in enumerate(self.rbms):
            best_loss = 121212121212.12
            since_improve = 0
            for j in tqdm(range(self.pretrain_iterations), desc='Pretraining Layer '+str(i+1)+' of '+str(len(self.rbms))):
                output = features[np.random.randint(0,features.shape[0],1)]
                for rbm in self.rbms[:i]:
                    output = rbm.sample_hidden_from_visible(output)
                loss = layer.rbn_update(output, self.learning_rbm_rate)
                if loss < best_loss:
                    best_loss = loss
                    since_improve = 0
                elif since_improve > 200:
                    self.learning_rbm_rate *= 0.96
                    since_improve = 0
                else:
                    since_improve += 1
           

    
    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
    
    def bias_variable(sefl,shape): #tf.Variable(tf.zeros([num_units]), name='bias')
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def rbn_layer(self,input_tensor, weights, bias, act, is_frozen,layer_name="rbn_layer"):
       # print("this",weights)
        with tf.name_scope(layer_name):
            with tf.name_scope('weights_rbn'):
                if is_frozen:
                    with tf.name_scope('frozen_weights'):
                        #weights_var = tf.Variable(initial_value=tf.zeros([weights.shape[0], weights.shape[1]], tf.float32),trainable=False, dtype=tf.float32,validate_shape=False)
                        #weights_new = tf.assign_add(weights_var, weights)
                        #self.variable_summaries(weights_new)
                        weights_new = tf.get_variable(layer_name+"weights_frozen",initializer=weights.astype(np.float32),trainable=False, dtype=tf.float32)
                else:
                    with tf.name_scope('not_frozen_weights'):
                        #weights_var = tf.Variable(initial_value=tf.zeros([weights.shape[0], weights.shape[1]], tf.float32), dtype=tf.float32,validate_shape=False)
                        #weights_new = tf.assign_add(weights_var, weights)
                        #self.variable_summaries(weights_new)
                        weights_new = tf.get_variable(layer_name+"_weights_not_frozen",initializer=weights.astype(np.float32), dtype=tf.float32)
            with tf.name_scope('biases'):
                if is_frozen:
                    with tf.name_scope('frozen_bias'):
                        #biases_var = tf.Variable(initial_value=tf.zeros([weights.shape[1]]),trainable=False, dtype=tf.float32, validate_shape=False)
                        #bias_new = tf.assign_add(biases_var, np.squeeze(bias))
                        #self.variable_summaries(bias_new)
                        bias_new = tf.get_variable(layer_name+"bias_frozen",initializer=bias.astype(np.float32),trainable=False, dtype=tf.float32)
                else:
                     with tf.name_scope('not_frozen_bias'):
                        #biases_var = tf.Variable(initial_value=tf.zeros([weights.shape[1]]), dtype=tf.float32, validate_shape=False)
                        #bias_new = tf.assign_add(biases_var, np.squeeze(bias))
                        #self.variable_summaries(bias_new)
                        bias_new = tf.get_variable(layer_name+"bias_not_frozen",initializer=bias.astype(np.float32), dtype=tf.float32)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights_new) + bias_new
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations
        
    def nn_layer(self,input_tensor, input_dim, output_dim, act=tf.nn.relu,batch_normalization=True,layer_name="nn_layer"):
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.variable_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights)
            with tf.variable_scope('biases'):
                #biases = self.bias_variable([output_dim])
                biases = tf.Variable(tf.zeros([output_dim]))
                self.variable_summaries(biases)
            with tf.variable_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            if batch_normalization:
                preactivate = tf.layers.batch_normalization(preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations, preactivate
    
    def model_inputs(self, input_size, output_size):
        with tf.name_scope('placeholders'):
            in_placeholder = tf.placeholder(tf.float32, [None, input_size], name='input')
            out_placeholder = tf.placeholder(tf.float32, [None, output_size], name='labels')
            dropout_placeholder = tf.placeholder(tf.float32, name='dropout')
            return in_placeholder, out_placeholder, dropout_placeholder
    
    def rbm_network(self,out, rbms):
        with tf.name_scope("rbm"):
            for rbm in rbms:
                    out = self.rbn_layer(input_tensor=out,
                                         weights=rbm.W, 
                                         bias=rbm.bias_hidden, 
                                         act=tf.nn.sigmoid, 
                                         is_frozen=rbm.is_frozen,
                                         layer_name=rbm.name)
            num_prev_outputs = rbms[-1].num_hidden
            return out, num_prev_outputs

    def dense_network(self, out, num_prev_outputs, fully_connected_layers, activation,batch_normalization, dropout):
        with tf.name_scope("dense"):
            for i, connected in enumerate(fully_connected_layers):
                out = tf.layers.dropout(out, dropout, name='dropout_'+str(i))
                out,_ = self.nn_layer(out, num_prev_outputs, connected, act=activation[i],batch_normalization=batch_normalization[i],layer_name="dense_layer_"+str(i))
               
                num_prev_outputs = connected
            return out, num_prev_outputs

    def last_layer(self, out, num_prev_outputs,output_size, dropout):
        out = tf.layers.dropout(out, dropout, name='dropout_before_last_layer')
        out,net = self.nn_layer(out, num_prev_outputs, output_size, act=tf.nn.softmax,layer_name="OutputLayer")
        return out, net


    def network(self, features,dropout, rbms,fully_connected_layers, activation, output_size,batch_normalization):
        with tf.variable_scope('model_architecture'):
            out_rbm, num_prev_outputs = self.rbm_network(features, rbms)
            out_dense, num_prev_outputs_one = self.dense_network(out_rbm, num_prev_outputs,fully_connected_layers, activation,batch_normalization, dropout)
            out, net = self.last_layer(out_dense, num_prev_outputs_one,output_size, dropout)
            return out, net
   

    def model_loss_accuracy(self,labels,out, net):
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=net))
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(out, 1),tf.argmax(labels, 1))
            with tf.name_scope('final_accuracy'):
                accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        return loss, accuracy, correct_prediction

    def model_opt(self, loss, learning_rate, beta1):
        t_vars = tf.trainable_variables()
        t_vars_list = [var for var in t_vars if var.name.startswith('model_architecture')]
        train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss, var_list=t_vars_list)
        #train_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list=t_vars_list)
        return train_opt
    
    def build_graph(self,rbn_data,input_size, output_size):
        self.pretrain(rbn_data)
        if self.graph is None:
            g = tf.Graph()
        else:
            g = self.graph
        with g.as_default():
            self.features, self.labels, self.dropout = self.model_inputs(input_size, output_size)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.out, self.net = self.network(self.features, self.dropout, self.rbms, self.dense_layers, self.dense_activation, output_size, self.batch_normalization)
            self.loss, self.accuracy, self.correct_prediction = self.model_loss_accuracy(self.labels, self.out, self.net)
            self.exp_learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                    self.global_step * self.batch_size ,
                                                    self.learning_rate_decay_examples,
                                                    self.learning_rate_decay,
                                                    staircase=True)
            self.train_op = self.model_opt(self.loss, self.exp_learning_rate,self.beta1)
        self.graph = g
       