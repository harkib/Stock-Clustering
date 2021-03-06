from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN, FeatureAgglomeration, MiniBatchKMeans, OPTICS, MeanShift
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import json
import os

class Clusters:

    def __init__(self, Y, Y_name):

        self.n_clusters = max(Y)+1
        self.clusters = dict(zip(range(self.n_clusters),[[] for i in range(self.n_clusters)]))
        self.n_stocks = len(Y_name)

        self.correlations_daily = {}
        self.correlations_weekly = {}
        self.correlations_avg = {}

        self.n_R_considered = 0
        self.n_clusters_occupied = 0
        self.n_stocks_considered = 0


        self.r_daily = 0
        self.r_weekly = 0
        self.r_avg = 0


        for y, stock in zip(Y,Y_name):
            
            if y < 0: # some algorithms give -1 classification if it cannot find a good cluster
                continue

            self.clusters[y].append(stock)
    
    def correlation(self, test_daily, test_weekly):

        test_daily = test_daily.transpose()
        test_weekly = test_weekly.transpose()

        # compute daily r
        r_sum = 0
        d_c = 0
        for key in self.clusters.keys():
            
            stocks = self.clusters[key]
            if len(stocks) > 1:
                d_c += 1
                n = len(stocks)
                rs = np.array(test_daily[stocks].corr())[np.triu_indices(n,k=1)]
                r_sum += np.sum(rs)
                self.correlations_daily[key] = np.average(rs)
                self.n_clusters_occupied += 1
                self.n_stocks_considered += n
                self.n_R_considered += (n**2 -n)/2

        r_daily = r_sum/self.n_R_considered

        # compute weekly r
        w_c = 0
        r_sum = 0
        for key in self.clusters.keys():
            
            stocks = self.clusters[key]
            if len(stocks) > 1:
                w_c += 1
                rs = np.array(test_weekly[stocks].corr())[np.triu_indices(len(stocks),k=1)]
                r_sum += np.sum(rs)
                self.correlations_weekly[key] = np.average(rs)
                self.correlations_avg[key] = (self.correlations_daily[key] + np.average(rs))/2

        r_weekly = r_sum/self.n_R_considered

        r_avg = (r_daily + r_weekly)/2

        return {'Avg-R': r_avg, 'Daily-R': r_daily, 'Weekly-R':r_weekly, 'Coverage': self.n_stocks_considered/self.n_stocks}

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
    # daily = pd.read_pickle('Data\daily.pkl').set_index('Stock', drop =True)
    # weekly = pd.read_pickle('Data\weekly.pkl').set_index('Stock', drop =True)
    # monthly = pd.read_pickle('Data\monthly.pkl').set_index('Stock', drop =True)
    # test_daily = pd.read_pickle(r'Data\test_daily.pkl').set_index('Stock', drop =True)
    # test_weekly = pd.read_pickle(r'Data\test_weekly.pkl').set_index('Stock', drop =True)
    # GICS = pd.read_csv('Data\GICS-wiki.csv',encoding='ANSI').set_index('Stock', drop =True)

    daily_path  = os.path.join('Data','daily.pkl')
    weekly_path = os.path.join('Data','weekly.pkl')
    monthly_path = os.path.join('Data','monthly.pkl')
    GICS_path    = os.path.join('Data','GICS-wiki.csv')


    daily       = pd.read_pickle(daily_path).set_index('Stock', drop =True)
    weekly      = pd.read_pickle(weekly_path).set_index('Stock', drop =True)
    monthly     = pd.read_pickle(monthly_path).set_index('Stock', drop =True)
    
    test_daily  = pd.read_pickle(r'Data/test_daily.pkl').set_index('Stock', drop =True)
    test_weekly = pd.read_pickle(r'Data/test_weekly.pkl').set_index('Stock', drop =True)

    # https://stackoverflow.com/questions/19699367/for-line-in-results-in-unicodedecodeerror-utf-8-codec-cant-decode-byte

    GICS = pd.read_csv(GICS_path, encoding='ISO-8859-1').set_index('Stock', drop =True)


    # normalize base data
    daily = normalize_df(daily)
    weekly = normalize_df(weekly)
    monthly = normalize_df(monthly)

    # create GICS features
    GICS_Sector = pd.get_dummies(GICS['GICS Sector'])
    GICS_Sub = pd.get_dummies(GICS['GICS Sub Industry'])

    # create dict of input datas
    dfs = { 'Daily'     : daily,
            'Weekly'    : weekly,
            'Monthly'   : monthly,
            'GICS_Sub'  : GICS_Sub[GICS_Sub.index.isin(daily.index)],
            'GICS_Sector'     : GICS_Sector[GICS_Sector.index.isin(daily.index)],
            'Daily+Weekly'    : daily.join(weekly,lsuffix='-d', rsuffix='-w'),
            'Daily+Monthly'   : daily.join(monthly,lsuffix='-d', rsuffix='-m'),
            'Weekly+Monthly'  : weekly.join(monthly,lsuffix='-w', rsuffix='-m'),
            'Daily+Weekly+Monthly'    : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(monthly,lsuffix='', rsuffix='-m'),
            'Daily+GICS_Sector'       : daily.join(GICS_Sector,how='inner'),
            'Weekly+GICS_Sector'      : weekly.join(GICS_Sector,how='inner'),
            'Monthly+GICS_Sector'     : monthly.join(GICS_Sector,how='inner'),
            'Daily+Weekly+GICS_Sector'      : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(GICS_Sector,how='inner'),
            'Daily+Monthly+GICS_Sector'     : daily.join(monthly,lsuffix='-d', rsuffix='-m').join(GICS_Sector,how='inner'),
            'Weekly+Monthly+GICS_Sector'    : weekly.join(monthly,lsuffix='-w', rsuffix='-m').join(GICS_Sector,how='inner'),
            'Daily+Weekly+Monthly+GICS_Sector'    : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(monthly,lsuffix='', rsuffix='-m').join(GICS_Sector,how='inner'),
            'Daily+GICS_Sub'       : daily.join(GICS_Sub,how='inner'),
            'Weekly+GICS_Sub'      : weekly.join(GICS_Sub,how='inner'),
            'Monthly+GICS_Sub'     : monthly.join(GICS_Sub,how='inner'),
            'Daily+Weekly+GICS_Sub'      : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(GICS_Sub,how='inner'),
            'Daily+Monthly+GICS_Sub'     : daily.join(monthly,lsuffix='-d', rsuffix='-m').join(GICS_Sub,how='inner'),
            'Weekly+Monthly+GICS_Sub'    : weekly.join(monthly,lsuffix='-w', rsuffix='-m').join(GICS_Sub,how='inner'),
            'Daily+Weekly+Monthly+GICS_Sub'    : daily.join(weekly,lsuffix='-d', rsuffix='-w').join(monthly,lsuffix='', rsuffix='-m').join(GICS_Sub,how='inner'),
    }

    # create dict of models 
    models = {
                'AgglomerativeClustering_100' : AgglomerativeClustering(n_clusters=100),
                'AgglomerativeClustering_150' : AgglomerativeClustering(n_clusters=150),
                'AgglomerativeClustering_200' : AgglomerativeClustering(n_clusters=200),
                'AgglomerativeClustering_250' : AgglomerativeClustering(n_clusters=250),
                'AgglomerativeClustering_300' : AgglomerativeClustering(n_clusters=300),
                'AgglomerativeClustering_350' : AgglomerativeClustering(n_clusters=350),
                'AgglomerativeClustering_400' : AgglomerativeClustering(n_clusters=400),
                'KMeans_100'                : KMeans(n_clusters=100), 
                'KMeans_150'                : KMeans(n_clusters=150), 
                'KMeans_200'                : KMeans(n_clusters=200), 
                'KMeans_250'                : KMeans(n_clusters=250), 
                'KMeans_300'                : KMeans(n_clusters=300), 
                'KMeans_350'                : KMeans(n_clusters=350), 
                'KMeans_400'                : KMeans(n_clusters=400), 
                'AffinityPropagation'   : AffinityPropagation(random_state=5),
                'DBSCAN_0_5'            : DBSCAN(eps=0.5, min_samples = 2),
                'DBSCAN_1'              : DBSCAN(eps=1,   min_samples = 2),
                'DBSCAN_1_25'           : DBSCAN(eps=1.25,min_samples = 2),
                'DBSCAN_1_5'            : DBSCAN(eps=1.5, min_samples = 2),
                'DBSCAN_2'              : DBSCAN(eps=2,   min_samples = 2),

                # FeatureAgglomeration did not have fit_predict and fail in this version
                # 'FeatureAgglomeration_100'   : FeatureAgglomeration(n_clusters=100),
                # 'FeatureAgglomeration_150'   : FeatureAgglomeration(n_clusters=150),
                # 'FeatureAgglomeration_200'   : FeatureAgglomeration(n_clusters=200),
                # 'FeatureAgglomeration_250'   : FeatureAgglomeration(n_clusters=250),
                # 'FeatureAgglomeration_300'   : FeatureAgglomeration(n_clusters=300),
                # 'FeatureAgglomeration_350'   : FeatureAgglomeration(n_clusters=350),
                # 'FeatureAgglomeration_400'   : FeatureAgglomeration(n_clusters=400),



                'MiniBatchKMeans_100'   : MiniBatchKMeans(n_clusters=100),
                'MiniBatchKMeans_150'   : MiniBatchKMeans(n_clusters=150),
                'MiniBatchKMeans_200'   : MiniBatchKMeans(n_clusters=200),
                'MiniBatchKMeans_250'   : MiniBatchKMeans(n_clusters=250),
                'MiniBatchKMeans_300'   : MiniBatchKMeans(n_clusters=300),

                # 'OPTICS_0_5'                :OPTICS(eps = 0.5, min_samples = 2),
                'OPTICS_1_0'                :OPTICS(eps = 1.5, min_samples = 2),
                # 'OPTICS_1_5'                :OPTICS(eps = 2.0, min_samples = 2),
                # 'OPTICS_2_5'                :OPTICS(eps = 2.5, min_samples = 2),
                # 'OPTICS_3_0'                :OPTICS(eps = 3.0, min_samples = 2),


                'MeanShift_1_0'                :MeanShift(bandwidth = 1.0),
                'MeanShift_1_5'                :MeanShift(bandwidth = 1.5),
                'MeanShift_2_0'                :MeanShift(bandwidth = 2.0),
                'MeanShift_2_5'                :MeanShift(bandwidth = 2.5),
                'MeanShift_3_0'                :MeanShift(bandwidth = 3.0),


    }



    # test all combinations
    results = []
    for model_key in models.keys():
        for df_key in dfs.keys():
            
            # AffinityPropagation cannot cluster GICS_Sub and GICS_Sector
            if (model_key =='AffinityPropagation') and (df_key == 'GICS_Sub' or df_key == 'GICS_Sector'):
                continue

            X = np.array(dfs[df_key])
            Y_name = dfs[df_key].index

            # build model
            n_clusters = 150
            model = models[model_key]
            Y = model.fit_predict(X)

            # evaluate model
            clusters = Clusters(Y,Y_name)
            result = clusters.correlation(test_daily, test_weekly)
            result['Data'] = df_key
            result['Model'] = model_key
            results.append(result)
            print(result)

            # save clusters data
            # with open('Output\\Clusters\\'+ model_key + '_' + df_key + '.json', 'w+') as fp:
            #     json.dump(clusters.clusters, fp)
            # with open('Output\\Correlations\\'+ model_key + '_' + df_key+ '.json', 'w+') as fp:
            #     json.dump(clusters.correlations_avg, fp)

            Clusters_path     = os.path.join('Output','Clusters')
            Correlations_path = os.path.join('Output','Correlations')

            with open(Clusters_path + model_key + '_' + df_key + '.json', 'w+') as fp:
                json.dump(clusters.clusters, fp)
            with open(Correlations_path + model_key + '_' + df_key+ '.json', 'w+') as fp:
                json.dump(clusters.correlations_avg, fp)

    # pd.DataFrame(results).to_pickle(r'Output\combination_results.pkl')
    result_path = os.path.join('Output','combination_results.pkl')

    # pd.DataFrame(results).to_pickle(r'Output/combination_results.pkl')
    pd.DataFrame(results).to_pickle(result_path)




