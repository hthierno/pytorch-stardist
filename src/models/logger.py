import os
from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt


class Logger:
    """
    Utility class to log metrics during training.
    """
    def __init__(self):
        self.info = dict()
        self.values_per_epoch = [] # Unnormalized
        self.batch_sizes_per_epoch = []
        
        self.values_per_step = []
        self.batch_sizes_per_step = []
    
    def log(self, name, value, epoch=None, batch_size=None, per_step=True, per_epoch=True):
        
        
        
        if name not in self.info:
            self.info[name] = {
                "values_per_epoch": [],
                "batch_sizes_per_epoch": [],
                "values_per_step": [],
                "batch_sizes_per_step": []
            }
        
        nb_epoch_logged = len( self.info[name]['values_per_epoch'] )
        
        if epoch is None:
            epoch = nb_epoch_logged
        
        assert epoch <= nb_epoch_logged, (name, epoch, nb_epoch_logged)
        
        if epoch == nb_epoch_logged:
            self.info[name]['values_per_epoch'].append(0)
            self.info[name]['batch_sizes_per_epoch'].append( 0 )
            self.info[name]['values_per_step'].append( [] )
        
        #cur_epoch_value = self.info[name]['values_per_epoch'][epoch]
        self.info[name]['batch_sizes_per_epoch'][epoch] += batch_size
        self.info[name]['values_per_epoch'][epoch] += batch_size * value
        #self.info[name]['batch_sizes_per_step'] 
        self.info[name]['values_per_step'][epoch].append(value)
    
    
    def get_mean_value(self, name, epoch=None):
        if epoch is None:
            epoch = -1
        
        if epoch >= len( self.info[name]['values_per_epoch'] ):
            return None
            
        n = self.info[name]['batch_sizes_per_epoch'][epoch]
        l = self.info[name]['values_per_epoch'][epoch]
        return l/n
    
    def _makedirs(self, path):
        path = Path(path)
        if path.suffix:
            path = path.parent
        if not os.path.exists(path):
            print(f"The path <{path}> doesn't exists. It will be created!")
            os.makedirs(path)

    def to_pickle(self, path="./logs/metrics_logs.pkl"):
        self._makedirs(path)
        with open(path, "wb") as f:
            pickle.dump(self.info, f)
        
        print(f"Logs saved to the pickle format at <{path}>")

    
    def load_pickle(self, path="./logs/metrics_logs.pkl", epoch=None):
        with open(path, "rb") as f:
            info = pickle.load(f)
        self.info = info

        if epoch is not None:
            for name in self.info.keys():
                self.info[name]['values_per_epoch']= self.info[name]['values_per_epoch'][:epoch]
                self.info[name]['batch_sizes_per_epoch'] = self.info[name]['batch_sizes_per_epoch'][:epoch]
                self.info[name]['values_per_step'] = self.info[name]['values_per_step'][:epoch]

        
    
    def _to_df(self):
        if len( self.info ) == 0:
            return pd.DataFrame()
            
        n_epochs = [ len( self.info[name]['values_per_epoch'] ) for name in self.info.keys() ]
        n_epochs_max = max( n_epochs )
        
        mean_values_per_epoch = {
            name: [ self.get_mean_value(name, epoch) \
                   for epoch in range( n_epochs_max ) ] 
            for name in self.info.keys()
        }
        
        
        return pd.DataFrame( mean_values_per_epoch )
    
    def to_csv(self, path="./logs/metrics_logs.csv"):
        self._makedirs(path)
        self._to_df().to_csv(path)
        print(f"Logs saved to the csv format at <{path}>")
    
    def plot(self, path="./logs/figures", figsize=((21, 15))):

        self._makedirs(path)
        
        if path is not None:
            path = Path( path )
        
        
        df = self._to_df()
        
        for name in df.columns:
            
            fig = plt.figure(figsize=figsize)
                
            plt.plot( df[name] )
            
            plt.savefig( path / f"{name.replace('/', '_')}.png" )
            
            plt.close(fig)