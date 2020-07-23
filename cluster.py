from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

class Clusters:

    clusters =  {}
    correlations = {}
    n_clusters = 0
    # n_stocks = 0

    info_n_stocks = 0
    

    def __init__(self,n_clusters_, Y, Y_name):

        self.n_clusters = n_clusters_
        self.clusters = dict(zip(range(n_clusters_),[[] for i in range(n_clusters_)]))

        for y, stock in zip(Y,Y_name):
            self.clusters[y].append(stock)
    
    def correlation(self, test_data):

        # self.n_stocks = len(test_data['Stock'])
        test_data = test_data.set_index('Stock', drop =True).transpose()
    
        r_sum = 0
        for key in self.clusters.keys():
            
            stocks = self.clusters[key]
            self.info_n_stocks += len(stocks)

            if len(stocks) > 1:
                rs = np.array(test_data[stocks].corr())[np.triu_indices(len(stocks),k=1)]
                r_avg = np.average(rs)
                r_sum += r_avg
                self.correlations[key] = r_avg

        return r_sum/self.n_clusters

    def print_(self,n=10):

        for key in self.clusters.keys():
            if len(self.clusters[key]) > 1:
                print(key, self.clusters[key][:n],self.correlations[key] )


    



if __name__ == '__main__':

    # Load Data 
    daily = pd.read_pickle('Data\daily.pkl')
    weekly = pd.read_pickle('Data\weekly.pkl')
    monthly = pd.read_pickle('Data\monthly.pkl')
    GICS = pd.read_csv('Data\GICS-wiki.csv',encoding='ANSI')
    test = pd.read_pickle('Data\\test.pkl')

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
    print('Average Cluster Correlation:', KMeans_Clusters.correlation(test))
    KMeans_Clusters.print_()


