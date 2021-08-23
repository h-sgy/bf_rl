#!/usr/bin/env python
# -*- coding: utf-8 -*-

from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO
import os
import tempfile
import zipfile
import pandas as pd
from datetime import timedelta, datetime, time, timezone
pd.options.display.max_columns = None
#pd.options.display.max_rows = None
pd.options.display.width = 2000
pd.options.display.float_format = '{:.3f}'.format

class Logger():

    def __init__(self):
        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)
        handler_format = Formatter('%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler = StreamHandler()
        stream_handler.setLevel(INFO)
        stream_handler.setFormatter(handler_format)
        self.logger.addHandler(stream_handler)

class GenerateData(object):

    def __init__(self, root_logger, input_dir):

        self.logger = root_logger
        self.timescale = "1s"
        self.input_dir = input_dir

        # 足データのカラムの設定
        self.columns = ['open','high','low','close','volume','buy_volume','sell_volume','exec_count','buy_exec_count','sell_exec_count','buy_value','sell_value','total_value','latency']
        self.columns2 = ['exec_date','side','price','size','id','latency']
        self.file_lines = 300000

    def run(self):

        # 入力データのディレクトリチェック
        if not os.path.exists(self.input_dir):
            self.logger.logger.error('Does not exist input directory: {}'.format(self.input_dir))
            exit(1)

        self.logger.logger.info('START generate ohlcv')
        self.logger.logger.info('input directory: {}'.format(self.input_dir))

        # 空のDataframeを作成
        summary_ohlc = pd.DataFrame(columns=['exec_date']+self.columns)
        summary_ohlc = summary_ohlc.set_index('exec_date')

        # 指定されたディレクトリからファイルを取得
        file_list = os.listdir(self.input_dir)
        assert(len(file_list)!=0 )

        load_first = True
        for file_name in file_list:
            if file_name.endswith('.csv') :
                # ディレクトリに存在するファイルを一つずつ読み込む
                df_executions = self.load_execution_data(file_name)
                # 秒足ohlcと出来高をを取得
                df_ohlc = self.generate_second_ohlc(df_executions)
                summary_ohlc = self.summarize_ohlc(summary_ohlc, df_ohlc)

            elif file_name.endswith('.zip') :
                # ディレクトリに存在するzipファイルを一つずつ読み込む
                df_executions = self.load_zipped_execution_data(file_name)
                # 秒足ohlcと出来高をを取得
                df_ohlc = self.generate_second_ohlc(df_executions)
                summary_ohlc = self.summarize_ohlc(summary_ohlc, df_ohlc)

            else:
                pass


        # 日を跨いだ足の統合
        self.logger.logger.info('summarizing candles ......')
        summary_ohlc = summary_ohlc.resample(self.timescale).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                     'volume': 'sum', 'buy_volume': 'sum', 'sell_volume': 'sum',
                                     'exec_count': 'sum', 'buy_exec_count': 'sum', 'sell_exec_count': 'sum',
                                     'buy_value': 'sum', 'sell_value': 'sum', 'total_value': 'sum', 'latency': 'mean'})

        jst = timezone(timedelta(hours=+9), 'JST')
        idx = pd.date_range(datetime.combine(min(summary_ohlc.index), time.min, tzinfo=jst), datetime.combine(max(summary_ohlc.index), time.max, tzinfo=jst), freq=self.timescale)
        summary_ohlc = summary_ohlc.reindex(idx)
        summary_ohlc = summary_ohlc.sort_index()

        return summary_ohlc
        # まとめたデータを日付で分ける
        summary_ohlc_list = self.separate_summary(summary_ohlc)

        
        # # 保存
        # for separate_summary in summary_ohlc_list:
        #     # 並べ替え
        #     separate_summary = separate_summary.sort_index()
        #     separate_summary = separate_summary[self.columns]

        #     self.save_ohlc_data(separate_summary)

    def executions(self):

        # 入力データのディレクトリチェック
        if not os.path.exists(self.input_dir):
            self.logger.logger.error('Does not exist input directory: {}'.format(self.input_dir))
            exit(1)

        self.logger.logger.info('START generate ohlcv')
        self.logger.logger.info('input directory: {}'.format(self.input_dir))

        # 空のDataframeを作成
        summary_ohlc = pd.DataFrame(columns=['exec_date']+self.columns2)
        summary_ohlc = summary_ohlc.set_index('exec_date')

        # 指定されたディレクトリからファイルを取得
        file_list = os.listdir(self.input_dir)
        assert(len(file_list)!=0 )

        load_first = True
        for file_name in file_list:
            if file_name.endswith('.csv') :
                # ディレクトリに存在するファイルを一つずつ読み込む
                df_executions = self.load_execution_data(file_name)
                summary_ohlc = self.summarize_ohlc(summary_ohlc, df_executions)

            elif file_name.endswith('.zip') :
                # ディレクトリに存在するzipファイルを一つずつ読み込む
                df_executions = self.load_zipped_execution_data(file_name)
                summary_ohlc = self.summarize_ohlc(summary_ohlc, df_executions)

            else:
                pass


        # 日を跨いだ足の統合
        self.logger.logger.info('summarizing candles ......')
        # summary_ohlc = summary_ohlc.resample('1S').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        #                              'volume': 'sum', 'buy_volume': 'sum', 'sell_volume': 'sum',
        #                              'exec_count': 'sum', 'buy_exec_count': 'sum', 'sell_exec_count': 'sum',
        #                              'buy_value': 'sum', 'sell_value': 'sum', 'total_value': 'sum'})
        summary_ohlc["exec_date"] = pd.to_datetime(summary_ohlc['exec_date'].replace('T', ' '))
        summary_ohlc = summary_ohlc.set_index('exec_date')
        summary_ohlc = summary_ohlc.sort_index()

        return summary_ohlc
        # まとめたデータを日付で分ける
        summary_ohlc_list = self.separate_summary(summary_ohlc)

        
        # # 保存
        # for separate_summary in summary_ohlc_list:
        #     # 並べ替え
        #     separate_summary = separate_summary.sort_index()
        #     separate_summary = separate_summary[self.columns]

        #     self.save_ohlc_data(separate_summary)

    def generate_second_ohlc(self, df_executions):

        df_executions["exec_date"] = pd.to_datetime(df_executions['exec_date'].replace('T', ' '))
        # 基本のOHLCVデータ作成
        df_ohlc_base = df_executions[["exec_date", "price"]].set_index('exec_date')
        df_ohlc = pd.concat([df_ohlc_base['price'].resample(self.timescale).ohlc()], axis=1)
        self.logger.logger.info('generated base ohlc data frame')

        # nan埋め
        # df_ohlc['close'] = df_ohlc['close'].fillna(method='ffill')
        # df_ohlc['open'] = df_ohlc['open'].fillna(df_ohlc['close'])
        # df_ohlc['high'] = df_ohlc['high'].fillna(df_ohlc['close'])
        # df_ohlc['low'] = df_ohlc['low'].fillna(df_ohlc['close'])
        # self.logger.logger.info('filled for nan')

        # TotalValue,Volumeのdataframe作成
        df_val = df_executions[['exec_date', 'price', 'size', 'side']]
        df_val = df_val.set_index('exec_date')
        df_val["total_value"] = df_val['price'] * df_val['size']

        buy_value = df_val.query('side == "BUY"').resample(self.timescale).sum()
        buy_value = buy_value.drop(columns=['price','size'])
        buy_value = buy_value.rename(columns={'total_value': 'buy_value'})
        
        sell_value = df_val.query('side == "SELL"').resample(self.timescale).sum()
        sell_value = sell_value.drop(columns=['price','size'])
        sell_value = sell_value.rename(columns={'total_value': 'sell_value'})

        df_val = df_val.resample(self.timescale).sum()
        df_val = df_val.drop(columns='price')
        df_val = df_val.rename(columns={'size': 'volume'})

        self.logger.logger.info('summarize volume and total_value')

        # 買い出来高、売り出来高のdataframe作成
        df_size = df_executions[['exec_date', 'side', 'size']]
        df_size = df_size.set_index('exec_date')
        buy_size = df_size.query('side == "BUY"').resample(self.timescale).sum()
        sell_size = df_size.query('side == "SELL"').resample(self.timescale).sum()
        self.logger.logger.info('summarize volume')

        # 約定回数のdataFrame作成
        df_count = df_executions[['exec_date', 'side']]
        df_count = df_count.set_index('exec_date')
        exec_count = df_count.resample(self.timescale).count()
        buy_exec_count = df_count.query('side == "BUY"').resample(self.timescale).count()
        sell_exec_count = df_count.query('side == "SELL"').resample(self.timescale).count()
        self.logger.logger.info('summarize counts')

        
        df_latency = df_executions[['exec_date', 'latency']]
        df_latency = df_latency.set_index('exec_date')
        latency = df_latency.resample(self.timescale).mean()

        # 基本OHLCVのデータフレームに列追加
        df_ohlc = pd.concat([df_ohlc, df_val], axis=1)
        df_ohlc = pd.concat([df_ohlc, buy_value], axis=1)
        df_ohlc = pd.concat([df_ohlc, sell_value], axis=1)
        df_ohlc = df_ohlc.join(buy_size)
        df_ohlc = df_ohlc.rename(columns={'size': 'buy_volume'})
        df_ohlc = df_ohlc.join(sell_size)
        df_ohlc = df_ohlc.rename(columns={'size': 'sell_volume'})
        df_ohlc = df_ohlc.join(exec_count)
        df_ohlc = df_ohlc.rename(columns={'side': 'exec_count'})
        df_ohlc = df_ohlc.join(buy_exec_count)
        df_ohlc = df_ohlc.rename(columns={'side': 'buy_exec_count'})
        df_ohlc = df_ohlc.join(sell_exec_count)
        df_ohlc = df_ohlc.rename(columns={'side': 'sell_exec_count'})
        df_ohlc = df_ohlc.join(latency)
        df_ohlc = df_ohlc.rename(columns={'side': 'latency'})
        self.logger.logger.info('append to base data frame')
        return df_ohlc

    def summarize_ohlc(self, summary_ohlc, df_ohlc):
        self.logger.logger.info('summarize_ohlc')
        summary_ohlc = pd.concat([summary_ohlc, df_ohlc], axis=0, sort=True)        # summary_ohlcとdf_ohlcを縦につなげる。
        self.logger.logger.info('summary lines: {}'.format(len(summary_ohlc)))
        return summary_ohlc

    def separate_summary(self, summary_ohlc):
        self.logger.logger.info('separate_summary')
        summary_ohlc_list = []

        day_ohlc = summary_ohlc.resample('1D').first()

        for i in range(len(day_ohlc.index.values)-1):
            tmp_summary = summary_ohlc.loc[day_ohlc.index.values[i]:day_ohlc.index.values[i+1]]
            summary_ohlc_list.append(tmp_summary.head(len(tmp_summary)-1))
            self.logger.logger.info('separated: {}'.format(day_ohlc.index.values[i]))
        summary_ohlc_list.append(summary_ohlc.loc[day_ohlc.index.values[-1]:])
        self.logger.logger.info('separated: {}'.format(day_ohlc.index.values[-1]))

        return summary_ohlc_list

    # def save_ohlc_data(self, df_ohlc):
    #     # ファイル名作成
    #     str_date = str(df_ohlc.index[-1])[:10]
    #     folder_name = '{}/'.format(self.output_dir)
    #     file_name = 'ohlc_{}'.format(str_date)
    #     with tempfile.TemporaryDirectory() as temp_path:
    #         df_ohlc.to_csv(temp_path+'/'+file_name+'.csv')        # 保存
    #         with zipfile.ZipFile(folder_name+file_name+'.zip', 'w') as zf:
    #             zf.write(temp_path+'/'+file_name+'.csv', arcname=file_name+'.csv', compress_type=zipfile.ZIP_DEFLATED)
    #         os.remove(temp_path+'/'+file_name+'.csv')
    #         self.logger.logger.info(' save on {}'.format(folder_name+file_name+'.zip'))

    def load_execution_data(self, file_name):
        input_path = os.path.join(self.input_dir, file_name)
        self.logger.logger.info( file_name )
        df_btc = pd.read_csv(input_path, header=None)
        df_btc.columns=['exec_date','side','price','size','id','latency']
        df_btc=df_btc.dropna(subset=['side'])
        self.logger.logger.info('Load : {}executions from [{}]'.format(len(df_btc),file_name))
        return df_btc

    def load_zipped_execution_data(self, file_name):
        input_path = os.path.join(self.input_dir, file_name)

        first = True
        with zipfile.ZipFile(input_path) as z:
            for arcname in z.namelist():
                if '.csv' in arcname :
                    self.logger.logger.info( arcname )
                    with z.open(arcname) as f:
                        if first :
                            df_btc = pd.read_csv(f, header=None)
                            first = False
                        else:
                            df_btc = df_btc.append( pd.read_csv(f, header=None) )

        df_btc.columns=['exec_date','side','price','size','id','latency']
        df_btc=df_btc.dropna(subset=['side'])
        self.logger.logger.info('Load : {}executions from [{}]'.format(len(df_btc),file_name))
        return df_btc


if __name__ == '__main__':
    logger = Logger()

    generate_ohlc = GenerateData( logger, './executions')
    generate_ohlc.run()
