# Stock-Clustering
Attempts to cluster the 505 stocks within the SP500 index into groups based on performance and public classifications

Install un-comman libraries:
- pip install yfinance
- pip install seaborn

View results:
1) run and view visualize.ipynb

Refetch data an recompute results:
1) python collectData.py
2) python cluster.py
3) run and view visualize.ipynb

Sample Clusters:

![test](https://github.com/harkib/Stock-Clustering/blob/master/Figures/Top_Model/CFG_CMA_FITB_HBAN_KEY_RF_ZION_DailyClose.png?raw=true)
![test](https://github.com/harkib/Stock-Clustering/blob/master/Figures/Top_Model/AAPL_ADBE_CRM_MSFT_PYPL_DailyClose.png?raw=true)
![test](https://github.com/harkib/Stock-Clustering/blob/master/Figures/Top_Model/CAT_CMI_DE_PCAR_DailyClose.png?raw=true)
![test](https://github.com/harkib/Stock-Clustering/blob/master/Figures/Top_Model/DHI_LEN_PHM_DailyClose.png?raw=true)
