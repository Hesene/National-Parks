
import pandas as pd
import numpy as np
import gc

from utils import one_hot_encoder

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
    df['day_month'] = df['day'].astype(str)+'_'+df['month'].astype(str)
    df['day_weekday'] = df['day'].astype(str)+'_'+df['weekday'].astype(str)
    df['day_weekofyear'] = df['day'].astype(str)+'_'+df['weekofyear'].astype(str)
    df['month_weekday'] = df['month'].astype(str)+'_'+df['weekday'].astype(str)
    df['month_weekofyear'] = df['month'].astype(str)+'_'+df['weekofyear'].astype(str)
    df['weekday_weekofyear'] = df['weekday'].astype(str)+'_'+df['weekofyear'].astype(str)

    df['park_day'] = df['park'].astype(str)+'_'+df['day'].astype(str)
    df['park_month'] = df['park'].astype(str)+'_'+df['month'].astype(str)
    df['park_weekday'] = df['park'].astype(str)+'_'+df['weekday'].astype(str)
    df['park_weekofyear'] = df['park'].astype(str)+'_'+df['weekofyear'].astype(str)

    # categorical変数を変換
    df, cat_cols = one_hot_encoder(df, nan_as_category=False)

    return df


if __name__ == '__main__':
    num_rows=10000
    # train & test
    df = train_test(num_rows)
    print(df)
