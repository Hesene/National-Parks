
import pandas as pd
import numpy as np
import gc

from utils import one_hot_encoder

################################################################################
# 提供データを読み込み、データに前処理を施し、モデルに入力が可能な状態でファイル出力するモジュール。
# get_train_dataやget_test_dataのように、学習用と評価用を分けて、前処理を行う関数を定義してください。
################################################################################

# Preprocess train.tsv and test.tsv
def train_test(num_rows=None):
    print("Loading datasets...")
    # load datasets
    train_df = pd.read_csv('../input/train.tsv', sep='\t', nrows=num_rows)
    test_df = pd.read_csv('../input/test.tsv', sep='\t', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    #testのtargetをnanにしときます
    test_df['visitors'] = np.nan

    # merge
    df = train_df.append(test_df[['datetime', 'park', 'visitors']]).reset_index()

    del train_df, test_df
    gc.collect()

    # 日付をdatetime型へ変換
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 季節性の特徴量を追加
    df['day'] = df['datetime'].dt.day.astype(object)
    df['month'] = df['datetime'].dt.month.astype(object)
    df['weekday'] = df['datetime'].dt.weekday.astype(object)
    df['weekofyear'] = df['datetime'].dt.weekofyear.astype(object)
#    df['day_month'] = df['day'].astype(str)+'_'+df['month'].astype(str)
#    df['day_weekday'] = df['day'].astype(str)+'_'+df['weekday'].astype(str)
#    df['day_weekofyear'] = df['day'].astype(str)+'_'+df['weekofyear'].astype(str)
#    df['month_weekday'] = df['month'].astype(str)+'_'+df['weekday'].astype(str)
#    df['month_weekofyear'] = df['month'].astype(str)+'_'+df['weekofyear'].astype(str)
#    df['weekday_weekofyear'] = df['weekday'].astype(str)+'_'+df['weekofyear'].astype(str)

#    df['park_day'] = df['park'].astype(str)+'_'+df['day'].astype(str)
#    df['park_month'] = df['park'].astype(str)+'_'+df['month'].astype(str)
#    df['park_weekday'] = df['park'].astype(str)+'_'+df['weekday'].astype(str)
#    df['park_weekofyear'] = df['park'].astype(str)+'_'+df['weekofyear'].astype(str)

    # categorical変数を変換
    df, cat_cols = one_hot_encoder(df, nan_as_category=False)

    return df

# Preprocess colopl.tsv
def colopl(num_rows=None):
    colopl = pd.read_csv('../input/colopl.tsv', sep='\t')

    return colopl

# Preprocess hotlink.tsv
def hotlink(num_rows=None):
    hotlink = pd.read_csv('../input/hotlink.tsv', sep='\t')

    return hotlink

# Preprocess nied_oyama.tsv
def nied_oyama(num_rows=None):
    nied_oyama = pd.read_csv('../input/nied_oyama.tsv', sep='\t')

    return nied_oyama

# Preprocess nightley.tsv
def nightley(num_rows=None):
    nightley = pd.read_csv('../input/nightley.tsv', sep='\t')
    nightley.loc[:,'datetime'] = pd.to_datetime(nightley['datetime'])

    # 1日先へシフト
    nightley[['Japan_count','Foreign_count']] = nightley[['Japan_count','Foreign_count']].shift()

    # additional feature
    nightley['NIGHTLEY_F_J_RATIO'] = nightley['Foreign_count'] / nightley['Japan_count']
    nightley['NIGHTLEY_F_J_SUM'] = nightley['Foreign_count'] + nightley['Japan_count']

    return nightley

# Preprocess weather.tsv
def weather(num_rows=None):
    weather = pd.read_csv('../input/weather.tsv', sep='\t')

    return weather

if __name__ == '__main__':
    num_rows=10000
    # train & test
    df = train_test(num_rows)

    # colopl
#    colopl = colopl(num_rows)
#    df = df.join(colopl, how='left', on=[['datetime', 'park']])

    # hotlink
#    hotlink = hotlink(num_rows)
#    df = df.join(hotlink, how='left', on=[['datetime', 'park']])

    # nightley
    nightley = nightley(num_rows)
    df = df.join(nightley, how='left', on='datetime')

    print(df)
