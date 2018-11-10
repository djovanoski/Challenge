import tensorflow as tf 
import numpy as np 
import pandas as pd 
from tensorflow.python.lib.io import file_io
from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from tensorflow.python.client import device_lib

from trainer import Trainer

FLAGS = flags.FLAGS

if __name__ == "__main__":
    #Dataset Path
    flags.DEFINE_string("path","sample.csv","Path to the csv file relative path from where is located this script")
    flags.DEFINE_float("size",0.15,"Train Split of the data ")
    flags.DEFINE_string("save_graph","result_challenge","Path for saving the graph definition ")
    #Hyperparameters
    flags.DEFINE_integer("pretrain", 10000,"Number of pretrain itteration using Restricted Boltzman Machines")
    #As TensorFlow in flags did not have passing list only integer , bool, float and strings i require string which is parsed later
    flags.DEFINE_string("rbm_layers","[128,128,256,256,128,256,256]","Create List of string after will be parsed of how much neurons you want per rbm layer, please use the brackets")
    flags.DEFINE_string("dense_layers","[256,128,64]","Create List of string after will be parsed of how much neurons you want per rbm layer, please use the brackets")

    flags.DEFINE_string("activation_rbm","sigmoid","Define as string activation function available in Tensorflow")
    flags.DEFINE_string("activation_dense","relu","Define as string activation function available in Tensorflow")
    flags.DEFINE_bool("freeze_rbms", True, "Define if you want during training the rbm trainable variables to be constant and not change during backprop")
    flags.DEFINE_bool("use_batch_normalization", True,"Bool if you want to use Batch Normalization Layer")
    flags.DEFINE_integer("batch_size", 32, "Define batch size")
    flags.DEFINE_float("learning_rate", 0.01, "Define learning rate")
    flags.DEFINE_float("beta1", 0.3, "Define beta1 which will be used by Adam Optimizer")
    flags.DEFINE_float("keep_chance", 0.8, "Define the keep chance for applying the dropout layer")
    
def parse_string(string_example):
    string_ = string_example.replace('[','').replace(']','').replace(' ', '').split(',')
    string_result = list(map(int, string_))
    return string_result

def find_class_by_name(name, modules):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def return_activation_list(activation_function, layers):
    activation_ = find_class_by_name(activation_function, [tf.nn])
    list_=[]
    for i in range(len(layers)):
        list_.append(activation_)
    return list_

def list_bool(bool_definition, layers):
    list_ = []
    if bool_definition:
        for i in range(len(layers)):
            list_.append(True)
    else:
        for i in range(len(layers)):
            list_.append(False)
    return list_


def main(unused_argv):
    logging.set_verbosity(tf.logging.INFO)
    logging.info("Tensorflow version: %s.", tf.__version__)
    rbm_layers = parse_string(FLAGS.rbm_layers)
    dense_layers = parse_string(FLAGS.dense_layers)
    activation_rbm = return_activation_list(FLAGS.activation_rbm, rbm_layers)
    activation_dense = return_activation_list(FLAGS.activation_dense, dense_layers)

    #freeze_rbms = list_bool(FLAGS.freeze_rbms, rbm_layers)
    freeze_rbms = [True, True, True, True, False, False, False]
    batch_normalization = list_bool(FLAGS.use_batch_normalization, dense_layers)
    output_activation = tf.nn.softmax
    trainer = Trainer(FLAGS.path, FLAGS.size,FLAGS.pretrain,rbm_layers,activation_rbm,freeze_rbms,
                      dense_layers,activation_dense,batch_normalization,output_activation,
                      FLAGS.batch_size,FLAGS.learning_rate,FLAGS.beta1,keep_chance=FLAGS.keep_chance)

    trainer.train(FLAGS.save_graph)
    

   

if __name__ == "__main__":
  app.run()