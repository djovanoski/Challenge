import tensorflow as tf 
import numpy as np 
import pandas as pd
from prepare_data import Dataset
from dbn import DeepBeliefNetwork
import os

from tensorflow import logging


class Trainer:
    def __init__(self, path, size,
                 pretrain_iterations,rbm_layers,rbm_activations,freeze_rbms,
                 dense_layers,dense_activations,batch_normalization,output_activation,
                 batch_size,learning_rate,beta1,keep_chance=0.5):

        self.dataset = Dataset(path, size)
        self.dbn = DeepBeliefNetwork(pretrain_iterations,rbm_layers,rbm_activations,freeze_rbms,
                                     dense_layers,dense_activations,batch_normalization,output_activation,
                                     batch_size,learning_rate,beta1,keep_chance=keep_chance)
        self.dbn.build_graph(self.dataset.rbn_data, self.dataset.input_size, self.dataset.output_size)

    def train(self, save_dir):
        
        epochs = 0
        since_improved = 0
        best_loss = 999999.0
        best_accuracy = 0
    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with self.dbn.graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()
                train_writer = tf.summary.FileWriter(save_dir + '/train', sess.graph)
                while since_improved < 15:
                    start= 0 
                    train_features, train_labels = self.unison_shuffle(self.dataset.train_features, self.dataset.train_labels)
                    while(start + self.dbn.batch_size < train_features.shape[0]):
                        batch_data, batch_labels = train_features[start:start+self.dbn.batch_size], train_labels[start:start+self.dbn.batch_size]
                        start += self.dbn.batch_size
                        _  = sess.run([self.dbn.train_op],feed_dict={self.dbn.features: batch_data,self.dbn.labels: batch_labels, self.dbn.dropout: self.dbn.keep_chance})
                    correct_prediction = 0 
                    loss_, accuracy_  = sess.run([self.dbn.loss, self.dbn.accuracy],feed_dict={self.dbn.features: self.dataset.validation_features,
                                                                                               self.dbn.labels: self.dataset.validation_labels, 
                                                                                               self.dbn.dropout: 1.0 })
                    acc_ = accuracy_ / self.dataset.validation_features.shape[0]
                  
                    if loss_ < best_loss:
                        best_loss = loss_
                        best_accuracy = acc_
                        since_improved = 0
                        self.saver.save(sess,os.path.join(save_dir, 'train.ckpt'),global_step=self.dbn.global_step)
                    else:
                        since_improved += 1

                    epochs += 1
                    logging.info(" Epoch: " + str(epochs)+ " Best Loss: " + ("%.4f" % best_loss) + " Validation Accuracy: " + ("%.4f" %  acc_))
            
            logging.info("Finised Training. Best Validation Accuracy is . " + str("%.4f" % best_accuracy))

    def unison_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

  