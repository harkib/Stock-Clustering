from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

class Clusters:

    clusters =  {}

    def __init__(self,n_clusters, Y, Y_name):
        
        self.clusters = dict(zip(range(n_clusters),[[] for i in range(n_clusters)]))

        for y, stock in zip(Y,Y_name):
            self.clusters[y].append(stock)
    
    def print_(self):

        for key in self.clusters.keys():
            if len(self.clusters[key]) > 1:
                print(key, self.clusters[key])



if __name__ == '__main__':

    # Load Data 
    daily = pd.read_pickle('Data\daily.pkl')
    weekly = pd.read_pickle('Data\weekly.pkl')
    monthly = pd.read_pickle('Data\monthly.pkl')
    GICS = pd.read_csv('Data\GICS-wiki.csv',encoding='ANSI')

    # define input
    x_cols = daily.columns[:-2]
    X = np.array(daily[x_cols])
    Y_stock = daily['Stock']

    # build model
    n_clusters = 100
    model = KMeans(n_clusters=n_clusters)
    Y = model.fit_predict(X)

    # evaluate model
    KMeans_Clusters = Clusters(n_clusters,Y,Y_stock)
    KMeans_Clusters.print_()




