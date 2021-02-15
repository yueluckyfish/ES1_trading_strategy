#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

class BacktestBase(object):
    
    def __init__(self,ticker1,ticker2):
        
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.signals = pd.DataFrame()
        self.portfolio = pd.DataFrame()
        self.signals[ticker1+'_close'] = None
        self.signals[ticker2+'_close'] = None
        self.signals['signal'] = None
        self.signals['position'] = None
        self.signals['portfolio'] = None
        
    def get_data(self,dataset):
        
        data=pd.ExcelFile(dataset)
        
        ticker1_data=pd.read_excel(data,self.ticker1,index_col='Date')
        ticker2_data=pd.read_excel(data,self.ticker2,index_col='Date')
        self.signals=pd.DataFrame(index=ticker1_data.index)
        
        self.signals[self.ticker1+'_close']=ticker1_data['LAST']
        self.signals[self.ticker2+'_close']=ticker2_data['LAST']
        self.signals=self.signals.dropna()
        
    

    def get_signal(self,window,n):
        #ticker1 is the underlying be traded
        def zscore(x, window):
            #get zscore for single time series
            r = x.rolling(window=window)
            m = r.mean().shift(1)
            s = r.std(ddof=0).shift(1)
            z = (x-m)/s
            return z
        self.signals['signal']=0.0
        self.signals[self.ticker1+'_zscore'] = zscore(self.signals[self.ticker1+'_close'], window)
        self.signals[self.ticker2+'_zscore'] = zscore(self.signals[self.ticker2+'_close'], window)
        self.signals[self.ticker1+'_pctchg'] = self.signals[self.ticker1+'_close'].pct_change().fillna(0.0)
        self.signals['signal'] = np.where(self.signals[self.ticker1+'_zscore'] > self.signals[self.ticker2+'_zscore'],1.0,0.0)
        self.signals['positions'] = self.signals['signal'].diff().fillna(0.0)
        diff_std = (self.signals[self.ticker1+'_zscore']- self.signals[self.ticker2+'_zscore']).std()

        it = iter(self.signals.index[:-1])
        for i in it:
            if self.signals['signal'][i]==1.0:
                if self.signals[self.ticker1+'_pctchg'][i]<=-0.5 or (self.signals[self.ticker1+'_zscore']-self.signals[self.ticker2+'_zscore'])[i]>(n*diff_std):
                    if len(self.signals[i:])>0:
                        newdf = self.signals[i:]
                        if len(newdf.index[newdf['positions']==-1.0])!=0:
                            nxt = newdf.index[newdf['positions']==-1.0][0]  
                            self.signals['positions'][i]=-1.0
                            self.signals['signal'][i:nxt]=0.0
                            self.signals['positions'][i]=-1.0
                            self.signals['positions'][nxt]=0.0
                            i = nxt
                            continue
                        continue

            try:
                i = next(it) 
            except StopIteration:
                break

    def plot_signals(self):
        fig = plt.figure(figsize=(25,10))
        ax1 = fig.add_subplot(111, ylabel='Price in $')
        ax1.plot(self.signals.index.map(mdates.date2num), self.signals[self.ticker1+'_zscore'])
        self.signals[[self.ticker1+'_zscore', self.ticker2+'_zscore']].plot(ax=ax1, lw=2.)
        ax1.plot(self.signals[self.signals.positions == 1.0].index,
                 self.signals[self.ticker1+'_zscore'][self.signals.positions == 1], "^", markersize=10, color="m")
        ax1.plot(self.signals[self.signals.positions == -1.0].index,
                 self.signals[self.ticker1+'_zscore'][self.signals.positions == -1], "v", markersize=10, color="k")
        plt.show()

        
    def backtest_portfolio(self):
    
        self.portfolio = pd.DataFrame(index=self.signals.index).fillna(0.0)
        self.portfolio['asset_ret'] = self.signals[self.ticker1+'_close'].fillna(0.0).pct_change()
        self.portfolio['returns'] = self.signals['signal'].shift(1).fillna(0.0)*self.portfolio['asset_ret']
        self.portfolio['cum_ret'] = self.portfolio['returns'].fillna(0.0).cumsum() 

    def get_results(self):
        annulized_ret=(self.portfolio['asset_ret']+1).cumprod()
        annulized_bm_ret=(self.portfolio['returns']+1).cumprod()
        ret = round(annulized_ret[-1]**(252/len(self.portfolio['asset_ret'].index))-1,3)
        bm_ret = round(annulized_bm_ret[-1]**(252/len(self.portfolio['returns'].index))-1,3)
        sharpe = round(np.sqrt(252)*self.portfolio['asset_ret'].mean()/self.portfolio['asset_ret'].std(),3)
        bm_sharpe = round(np.sqrt(252)*self.portfolio['returns'].mean()/self.portfolio['returns'].std(),3)
        out_df=pd.DataFrame(index=['Benchmark','Trading Strategy'])
        out_df['Annulized Return']=[ret,bm_ret]
        out_df['Annulized Sharpe']=[sharpe,bm_sharpe]
        print(out_df)

    
if __name__ == '__main__':
    
        base_test = BacktestBase('ES1','UX1')
        base_test.get_data("vix_data.xlsx")
        base_test.get_signal(47,2)
        base_test.plot_signals()
        base_test.backtest_portfolio()
        base_test.get_results()


# In[ ]:




