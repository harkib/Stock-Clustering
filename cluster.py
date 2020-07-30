from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import pandas as pd
import numpy as np
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
 
    

    def __init__(self, Y, Y_name):

        self.n_clusters = max(Y)+1
        self.clusters = dict(zip(range(self.n_clusters),[[] for i in range(self.n_clusters)]))

        for y, stock in zip(Y,Y_name):
            if y < 0: # some algorithms give -1 classification if it cannot find a good cluster
                continue
            self.clusters[y].append(stock)
    
    def correlation(self, test_daily, test_weekly):

        test_daily = test_daily.transpose()
        test_weekly = test_weekly.transpose()

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

# normalize cols of pd.df
def normalize_df(df):

    scaler = MinMaxScaler()
    X = np.array(df)
    scaler.fit(X)
    X = scaler.transform(X)

    return pd.DataFrame(X, columns = df.columns, index = df.index)



if __name__ == '__main__':

    # Load Data 
    daily = pd.read_pickle('Data\daily.pkl').set_index('Stock', drop =True)
    weekly = pd.read_pickle('Data\weekly.pkl').set_index('Stock', drop =True)
    monthly = pd.read_pickle('Data\monthly.pkl').set_index('Stock', drop =True)
    test_daily = pd.read_pickle(r'Data\test_daily.pkl').set_index('Stock', drop =True)
    test_weekly = pd.read_pickle(r'Data\test_weekly.pkl').set_index('Stock', drop =True)
    GICS = pd.read_csv('Data\GICS-wiki.csv',encoding='ANSI').set_index('Stock', drop =True)

    # normalize base data
    daily = normalize_df(daily)
    weekly = normalize_df(weekly)
    monthly = normalize_df(monthly)

    # create GICS features
    GICS_Sector = pd.get_dummies(GICS['GICS Sector'])
    GICS_Sub = pd.get_dummies(GICS['GICS Sub Industry'])

    # create dict of input datas
    dfs = { 'Daily'     : daily,
            # 'Weekly'    : weekly,
            # 'Monthly'   : monthly,
            # 'Daily + Weekly'    : daily.join(weekly,lsuffix='-d', rsuffix='-w'),
            # 'Daily + Monthly'   : daily.join(monthly,lsuffix='-d', rsuffix='-m'),
            # 'Weekly + Monthly'  : weekly.join(monthly,lsuffix='-w', rsuffix='-m'),
            # 'Daily + Weekly + Monthly'  : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(monthly,lsuffix='', rsuffix='-m'),
            # 'Daily + GICS_Sector'       : daily.join(GICS_Sector,how='inner'),
            # 'Weekly + GICS_Sector'      : weekly.join(GICS_Sector,how='inner'),
            # 'Monthly + GICS_Sector'     : monthly.join(GICS_Sector,how='inner'),
            # 'Daily + Weekly + GICS_Sector'      : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(GICS_Sector,how='inner'),
            # 'Daily + Monthly + GICS_Sector'     : daily.join(monthly,lsuffix='-d', rsuffix='-m').join(GICS_Sector,how='inner'),
            # 'Weekly + Monthly + GICS_Sector'    : weekly.join(monthly,lsuffix='-w', rsuffix='-m').join(GICS_Sector,how='inner'),
            # 'Daily + Weekly + Monthly + GICS_Sector'    : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(monthly,lsuffix='', rsuffix='-m').join(GICS_Sector,how='inner'),
            # 'Daily + GICS_Sub'       : daily.join(GICS_Sub,how='inner'),
            # 'Weekly + GICS_Sub'      : weekly.join(GICS_Sub,how='inner'),
            # 'Monthly + GICS_Sub'     : monthly.join(GICS_Sub,how='inner'),
            # 'Daily + Weekly + GICS_Sub'      : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(GICS_Sub,how='inner'),
            # 'Daily + Monthly + GICS_Sub'     : daily.join(monthly,lsuffix='-d', rsuffix='-m').join(GICS_Sub,how='inner'),
            # 'Weekly + Monthly + GICS_Sub'    : weekly.join(monthly,lsuffix='-w', rsuffix='-m').join(GICS_Sub,how='inner'),
            # 'Daily + Weekly + Monthly + GICS_Sub'    : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(monthly,lsuffix='', rsuffix='-m').join(GICS_Sub,how='inner'),
    }

    # create dict of models 
    n_clusters = 150
    models = {'AgglomerativeClustering' : AgglomerativeClustering(n_clusters=n_clusters),
                'KMeans'                : KMeans(n_clusters=n_clusters), 
                'AffinityPropagation'   : AffinityPropagation(random_state=5),
                'DBSCAN'                : DBSCAN(eps=.5,min_samples = 2),
    }



    # test all combinations
    for model_key in models.keys():
        for df_key in dfs.keys():

            X = np.array(dfs[df_key])
            Y_name = dfs[df_key].index

            # build model
            n_clusters = 150
            model = models[model_key]
            # model = AgglomerativeClustering(n_clusters=n_clusters)
            Y = model.fit_predict(X)

            # evaluate model
            clusters = Clusters(Y,Y_name)
            print(model_key, df_key, 'Cluster Correlation:')
            print(clusters.correlation(test_daily, test_weekly))
            # KMeans_Clusters.print_()


