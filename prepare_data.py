import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import logging

class Dataset:
    def __init__(self, path, size, preprocess_immediatelly=True):
        if preprocess_immediatelly:
            self.train_features,self.train_labels, self.validation_features, self.validation_labels, self.rbn_data = self.split_dataset(path, size)
            self.input_size = self.train_features.shape[1]
            self.output_size = self.train_labels.shape[1]
            logging.info("Data is prepared......")
        else:# for using the class with out preprocessing on init
            pass

    def split_dataset(self,path, size=0.2):
        '''Method for spliting the dataset on train and validation but also randomly pick equal amount of data for pretraining of weights'''
        features, one_hot_labels, scaled_features = self.create_numpy(path)
        train_features, validation_features, train_labels, validation_labels = train_test_split(features, one_hot_labels, test_size=size, random_state=42)
        rbn_data = self.create_for_rbn_data(train_features, train_labels)
        one_hot_train_label = LabelBinarizer().fit_transform(train_labels)
        one_hot_valid_label = LabelBinarizer().fit_transform(validation_labels)
        return train_features, one_hot_train_label, validation_features, one_hot_valid_label,rbn_data

    def create_for_rbn_data(self, features, labels):
        '''Create separate dataset from train dataset which will be used for pretraining the weights '''
        df = pd.DataFrame(np.concatenate((features, labels), axis=1))
        list_targets = list(np.unique(df.iloc[:,-1].values))
        idx = []
        for t in list_targets:
            see_if_work = list(np.random.permutation(df.loc[df[295] == t].index.values)[:866])
            idx += see_if_work
        new_df = df.iloc[idx,:]
        new_df_clean = new_df.drop(columns=[295], axis=1)
        return new_df_clean.values.astype(np.float32)

    def create_numpy(self,path):
        features, labels, scaled_features = self.read_process_csv(path)
        return features.values, labels, scaled_features

    def read_process_csv(self, path, normalize=True):
        list_names =  []
        for i in range(296):
            column = 'col'+str(i)
            list_names.append(column)
        df = pd.read_csv(path, names=list_names)
        numeric, categorical, one_only = self.get_list_columns(df)
        df_target = df.select_dtypes(include = ['object'])
        #df = df.drop(list(one_only), axis=1)
        df = df.drop(list(df_target.columns), axis=1)
        numeric.remove(list(df_target)[0])
        #df_stats = statistics(df, numeric)
        scaled_features = {}
        if normalize:
            for col in numeric:
                min_, max_ = df[col].min(axis=0),df[col].max(axis=0)
                scaled_features[col] = [min_, max_]
                df.loc[:,col] = (df[col] - min_)/(max_ - min_)
        return df, df_target, scaled_features

    def get_list_columns(self,df):
        list_numeric=[]
        list_categorical = []
        list_one_only = []
        for col in list(df.columns):
            if len(np.unique(df[col].values)) > 2:
                list_numeric.append(str(col))
            elif len(np.unique(df[col].values)) == 1:
                list_one_only.append(str(col))
            else:
                list_categorical.append(str(col))
        return list_numeric, list_categorical, list_one_only
    
