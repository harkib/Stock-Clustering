from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

class Clusters:

    clusters =  {}
    n_clusters = 0

    correlations_daily = {}
    correlations_weekly = {}

    n_clusters_occupied_daily = 0
    n_clusters_occupied_weekly = 0

    r_daily = 0
    r_weekly = 0
    r_avg = 0
 
    

    def __init__(self,n_clusters_, Y, Y_name):

        self.n_clusters = n_clusters_
        self.clusters = dict(zip(range(n_clusters_),[[] for i in range(n_clusters_)]))

        for y, stock in zip(Y,Y_name):
            self.clusters[y].append(stock)
    
    def correlation(self, test_daily, test_weekly):

        test_daily = test_daily.set_index('Stock', drop =True).transpose()
        test_weekly = test_weekly.set_index('Stock', drop =True).transpose()

        # compute daily r
        r_sum = 0
        for key in self.clusters.keys():
            
            stocks = self.clusters[key]
            if len(stocks) > 1:
                rs = np.array(test_daily[stocks].corr())[np.triu_indices(len(stocks),k=1)]
                r_avg = np.average(rs)
                r_sum += r_avg
                self.correlations_daily[key] = r_avg
                self.n_clusters_occupied_daily += 1

        r_daily = r_sum/self.n_clusters_occupied_daily

        # compute weekly r
        r_sum = 0
        for key in self.clusters.keys():
            
            stocks = self.clusters[key]
            if len(stocks) > 1:
                rs = np.array(test_weekly[stocks].corr())[np.triu_indices(len(stocks),k=1)]
                r_avg = np.average(rs)
                r_sum += r_avg
                self.correlations_weekly[key] = r_avg
                self.n_clusters_occupied_weekly += 1

        r_weekly = r_sum/self.n_clusters_occupied_weekly

        r_avg = (r_daily + r_weekly)/2
        
        return {'Avg': r_avg, 'Daily': r_daily, 'Weekly':r_weekly}

    def print_(self,n=10):

        for key in self.clusters.keys():
            if len(self.clusters[key]) > 1:
                print(key, self.clusters[key][:n],'Corr:',self.correlations_daily[key] )


    



if __name__ == '__main__':

    # Load Data 
    daily = pd.read_pickle('Data\daily.pkl')
    weekly = pd.read_pickle('Data\weekly.pkl')
    monthly = pd.read_pickle('Data\monthly.pkl')
    GICS = pd.read_csv('Data\GICS-wiki.csv',encoding='ANSI')
    test_daily = pd.read_pickle(r'Data\test_daily.pkl')
    test_weekly = pd.read_pickle(r'Data\test_weekly.pkl')

    # define input
    x_cols = daily.columns[:-2]
    X = np.array(daily[x_cols])
    Y_stock = daily['Stock']

    # build model
    n_clusters = 100
    model = AgglomerativeClustering(n_clusters=n_clusters)
    Y = model.fit_predict(X)

    # evaluate model
    KMeans_Clusters = Clusters(n_clusters,Y,Y_stock)
    print('Cluster Correlation:', KMeans_Clusters.correlation(test_daily, test_weekly))
    KMeans_Clusters.print_()


