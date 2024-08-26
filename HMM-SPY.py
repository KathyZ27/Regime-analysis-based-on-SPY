import time
import os
import psutil
import warnings
import numpy as np
import backtrader as bt
import pandas as pd
import datetime
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers

from regime_hmm_train import hmm_train, hmm_predict

# Define the strategy
class HMM(bt.Strategy): 
    alias = ('HMM',)

    def __init__(self, frequency=5, warm_up=1200):
        self.hmm_model = []
        self.reverse = []
        self.buysig = None
        self.frequency = frequency
        self.state = None
        self.warm_up = warm_up
        self.weights = 0
        self.values = []        

    def next(self):
        i = len(self.data0)

        # Skip the warm up period
        if i > self.warm_up:
            #if i % self.frequency == 1:
                # Get data for retraining
                # X = pd.DataFrame([self.data0.returns.get(ago=-1, size=self.warm_up)
                #                     ,self.data0.vix.get(ago=-1, size=self.warm_up)]).T
            X = np.array(self.data0.returns.get(ago=-1, size=self.warm_up)).reshape(-1,1)
                                
            #print(X.shape)
            # Retrain the hmm model
            hmm_model, reverse = hmm_train(X, M=30, rs=27)
            if len(self.hmm_model) < 5:
                self.hmm_model = [hmm_model] + self.hmm_model
                self.reverse = [reverse] + self.reverse
            else:
                self.hmm_model = [hmm_model] + self.hmm_model[:-1]
                self.reverse = [reverse] + self.reverse[:-1]
            
            # Update state
            self.state = [hmm_predict(model,
                                     np.array(self.data0.returns.get(ago=0, size=self.warm_up+1+((i-1)%self.frequency))).reshape(-1,1))
                          for model in self.hmm_model]
            self.state = [1-self.state[i] if self.reverse[i] else self.state[i] for i in range(len(self.state))]
            
            # buysig is a boolean. True means bull market and to buy.
            previous = self.buysig
            self.buysig = 1 if sum(self.state)/len(self.state) > 0.5 else 0
            
            if self.buysig != previous:
                self.close(data=self.data0)
                if self.buysig == 1:
                    size = int(self.broker.get_cash()/self.data0.lines.close[0])
                    self.buy(data=self.data0, size=size)
#                     self.order_target_percent(data=self.data0, target=1.0)
                #else:
                    #self.sell(data=self.data0, size=50)
                    #self.order_target_percent(data=self.data0, target=-1.0)
            

            
            # Record the portfolio value
            self.values.append(self.broker.getvalue())
                
                
    def stop(self):
        self.close(data=self.data0)
        returns = np.diff(self.values) / self.values[:-1]
        benchmark = np.array(self.data0.returns.get(ago=0, size=(len(self.values)-1)))
        r = np.array(self.data0.dgs10.get(ago=0, size=(len(self.values)-1)))
        
        # Mean, Std 
        mean = np.mean(returns) * np.sqrt(252) 
        std = np.std(returns) * np.sqrt(252) 
        # Downside risk
        downside_diff = np.minimum(returns-r,0)
        downside_risk = np.sqrt(np.mean(downside_diff **2))
        # Sharpe Ratio
        sharpe_ratio = (np.mean(returns)-np.mean(r)) / np.std(returns)
        sharpe_ratio *= np.sqrt(252)                    # Annualize
        
        ## Benchmark
        # Mean, Std 
        spy_mean = np.mean(benchmark) * np.sqrt(252) 
        spy_std = np.std(benchmark) * np.sqrt(252) 
        # Downside risk
        spy_downside_diff = np.minimum(benchmark-r,0)
        spy_downside_risk = np.sqrt(np.mean(spy_downside_diff **2))
        # Sharpe Ratio
        spy_sharpe_ratio = (np.mean(benchmark)-np.mean(r)) / np.std(benchmark)
        spy_sharpe_ratio *= np.sqrt(252)                    # Annualize
        
        
        print('Average Daily Return (Annualized): ', mean)
        print('Std (Annualized): ' , std)
        print('Downside risk: ' , downside_risk)
        print('Sharpe Ratio: ' , sharpe_ratio)
        
        print('Benchmark Average Daily Return (Annualized): ', spy_mean)
        print('Benchmark Std (Annualized): ' , spy_std)
        print('Benchmark Downside risk: ' , spy_downside_risk)
        print('Benchmark Sharpe Ratio: ' , spy_sharpe_ratio)
        
        pd.DataFrame([self.values
                      , self.data0.lines.close.get(ago=0, size=(len(self.values)))]).to_csv("Result.csv", index=False)
                    
                    
class My_CSVData(bt.feeds.PandasData):
    lines = ('returns','dgs10')
    params = (
        ('fromdate', datetime.datetime(2014, 1, 1)),
        ('todate', datetime.datetime(2023, 1, 1)),
        ('nullvalue', 0.0),
        ('dtformat', ("%Y-%m-%d")),
        ('datetime', 0),
        ('time', -1),
        ('high', 2),
        ('low', 3),
        ('open', 1),
        ('close', 4),
        ('volume', 6),
        ('openinterest', -1),
        ('returns', 7),
        ('dgs10',8),
    )


if __name__ == '__main__':
    start_date = datetime.datetime(2014, 8, 1)
    end_date = datetime.datetime(2024, 1, 1)
    warnings.filterwarnings("ignore")
    
    # YOUR FILE PATH HERE!
    filepath = r"AssetData.csv"


    # initialize the cerebro engine
    cerebro = bt.Cerebro()
    
    # Add data
    rut = pd.read_csv(filepath, index_col = None, parse_dates=['Date'])
    data_feed = My_CSVData(dataname=rut)
    cerebro.adddata(data_feed, name="RUT")

    # Add strategy
    cerebro.addstrategy(HMM)
    
    # Set cash 
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0)
    
    # Print starting cash
    print('Starting Portfolio Value: $%.2f' % cerebro.broker.getvalue())
    
    # Run the strategy
    cerebro.run()
    
    print('Ending Portfolio Value: $%.2f' % cerebro.broker.getvalue())

    # Plot the results
    cerebro.plot(iplot=False)
