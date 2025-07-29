import numpy as np
import pandas as pd


### 根据因子生成买卖信号以及相关辅助参数


class SignalHandler:
    def __init__(self,factor_column = None,long_range = None,short_range = None,multi_feature_filter = False,**kwargs):
        self.feature_columns = kwargs.get("feature_columns",None)
        self.long_thresholds = kwargs.get("long_thresholds",None) # 大于阈值做多
        self.short_thresholds = kwargs.get("short_thresholds",None) #小于阈值做空


        self.factor_column = factor_column
        self.factor_long_threshold = long_range
        self.factor_short_threshold = short_range
        self.multi_feature_filter = multi_feature_filter
    def generate_signal(self,df:pd.DataFrame):
        """
        为对应的df生成信号
        return:生成的信号series
        """

        df['signal'][df[self.factor_column]<self.factor_short_threshold] = -1
        df['signal'][df[self.factor_column]>self.factor_long_threshold] = 1

        if self.multi_feature_filter:
            df['long_signal'] = 1
            df['short_signal'] = -1
            for column in self.feature_columns:
                df['long_signal'][df[column] < self.long_thresholds[column]] = 0
                df['short_signal'][df[column] > self.short_thresholds[column]] = 0
            long_signal = np.where(df['long_signal'] == 1 & df['signal'] == 1, 1, 0)
            short_signal = np.where(df['short_signal'] == -1 & df['signal'] == -1, -1, 0)
            df['signal'] = np.where(long_signal == 1, 1, short_signal)
        return df['signal']

    def compute_atr(self,df:pd.DataFrame,period = 14)-> pd.Series:
        tr = self.compute_true_range(df)
        atr = tr.rolling(period).mean()
        return atr
    
    @staticmethod
    def compute_true_range(df:pd.DataFrame)-> pd.Series:
        prev_close = df['close'].shift(1)

        tr = pd.concat([
            df['high']-df['low'],
            (df["high"]-prev_close).abs(),
            (df['low']-prev_close).abs()
        ],axis=1).max(axis=1)

        return tr

