
import pandas as pd
import numpy as np
import gc

from jpholiday import is_holiday

from utils import one_hot_encoder, PARK_POINT, PARKS

################################################################################
# 提供データを読み込み、データに前処理を施し、モデルに入力が可能な状態でファイル出力するモジュール。
# get_train_dataやget_test_dataのように、学習用と評価用を分けて、前処理を行う関数を定義してください。
################################################################################

# 日本の休日
def getJapaneseHolidays(dates):
    japanese_holiday = dates.dt.date.apply(is_holiday).astype(int)

    # 祝日データに土日を追加
    japanese_holiday += (dates.dt.weekday==5).astype(int)
    japanese_holiday += (dates.dt.weekday==6).astype(int)

    # 年末年始の6日間を休日に変更
    japanese_holiday += ((dates.dt.month==12)&(dates.dt.day==28)&(japanese_holiday==0)).astype(int)
    japanese_holiday += ((dates.dt.month==12)&(dates.dt.day==29)&(japanese_holiday==0)).astype(int)
    japanese_holiday += ((dates.dt.month==12)&(dates.dt.day==30)&(japanese_holiday==0)).astype(int)
    japanese_holiday += ((dates.dt.month==12)&(dates.dt.day==31)&(japanese_holiday==0)).astype(int)

    japanese_holiday += ((dates.dt.month==1)&(dates.dt.day==1)&(japanese_holiday==0)).astype(int)
    japanese_holiday += ((dates.dt.month==1)&(dates.dt.day==2)&(japanese_holiday==0)).astype(int)
    japanese_holiday += ((dates.dt.month==1)&(dates.dt.day==3)&(japanese_holiday==0)).astype(int)
    japanese_holiday += ((dates.dt.month==1)&(dates.dt.day==4)&(japanese_holiday==0)).astype(int)

    return japanese_holiday

# 休みの谷間の平日を休日として埋める
def fillHolidays(holidays):
    for i, h in enumerate(holidays):
        if h==0:
            if holidays[i-1]==1 & holidays[i+1]==1:
                holidays[i]==1
    return holidays

# 連休数
def getNumHolidays(holidays):
    holiday1 = [holidays[0]]
    holiday2 = [holidays[-1]]
    for i, h in enumerate(holidays[1:]):
        if h==0:
            holiday1.append(0)
        else:
            holiday1.append(holiday1[i]+h)

    for i, h in enumerate(list(reversed(holidays))[1:]):
        if h==0:
            holiday2.append(0)
        else:
            holiday2.append(holiday2[i]+h)

    np.array(holiday1)+np.array(list(reversed(holiday2)))-1

    numholidays =pd.Series(np.array(holiday1)+
                           np.array(list(reversed(holiday2)))-1,
                           index=holidays.index).replace(-1,0)

    return numholidays

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

    # 日本の祝日データを追加
    df['japanese_holiday'] = getJapaneseHolidays(df['datetime'])

    # 連休数のファクターを生成
    holidays = df.groupby('datetime')['japanese_holiday'].mean().replace(2,1)
    holidays = fillHolidays(holidays) # 休日の谷間の平日を休日にする
    df['num_holidays'] = df['datetime'].map(getNumHolidays(holidays))

    # 季節性の特徴量を追加
    df['day'] = df['datetime'].dt.day.astype(object)
    df['month'] = df['datetime'].dt.month.astype(object)
    df['weekday'] = df['datetime'].dt.weekday.astype(object)
    df['weekofyear'] = df['datetime'].dt.weekofyear.astype(object)
#    df['day_month'] = df['day'].astype(str)+'_'+df['month'].astype(str)
    df['day_weekday'] = df['day'].astype(str)+'_'+df['weekday'].astype(str)
#    df['day_weekofyear'] = df['day'].astype(str)+'_'+df['weekofyear'].astype(str)
    df['month_weekday'] = df['month'].astype(str)+'_'+df['weekday'].astype(str)
    df['month_weekofyear'] = df['month'].astype(str)+'_'+df['weekofyear'].astype(str)
#    df['weekday_weekofyear'] = df['weekday'].astype(str)+'_'+df['weekofyear'].astype(str)

    df['park_day'] = df['park'].astype(str)+'_'+df['day'].astype(str)
    df['park_month'] = df['park'].astype(str)+'_'+df['month'].astype(str)
    df['park_weekday'] = df['park'].astype(str)+'_'+df['weekday'].astype(str)
    df['park_japanese_holiday'] = df['park'].astype(str)+'_'+df['japanese_holiday'].astype(str)
#    df['park_weekofyear'] = df['park'].astype(str)+'_'+df['weekofyear'].astype(str)

    # categorical変数を変換
    df_res, cat_cols = one_hot_encoder(df, nan_as_category=False)

    # stratify & mearge用
    df_res['park'] = df['park']
    df_res['weekofyear'] = df['weekofyear'].astype(int)
    df_res['weekday'] = df['weekday'].astype(int)
    df_res['year'] = df['datetime'].dt.year.astype(int)
    df_res['month'] = df['datetime'].dt.month.astype(int)
    df_res['park_month'], _ = pd.factorize(df['park_month'])
    df_res['ISESHIMA_summit'] = ((df['park']=='伊勢志摩国立公園')&df['japanese_holiday']&('2016-5-27'>df['datetime'])&(df['datetime']>'2015-6-5')).astype(int) # 2016年伊勢島サミット開催決定後の休日フラグ

    return df_res

# Preprocess colopl.tsv
def colopl(num_rows=None):
    colopl = pd.read_csv('../input/colopl.tsv', sep='\t')

    # 1-9をとりあえず5で埋めます
    colopl['count'] = colopl['count'].replace('1-9', 5).astype(int)

    # 月ごとに集計
    colopl = colopl.pivot_table(index=['year', 'month'], columns='country_jp', values='count', aggfunc=sum)

    #　１ヶ月先へシフト
    colopl = colopl.shift()

    # カラム名を変更
    colopl.columns = ['COLOPL_'+ c for c in colopl.columns]

    return colopl

# Preprocess hotlink.tsv
def hotlink(num_rows=None):
    # load csv
    hotlink = pd.read_csv('../input/hotlink.tsv', sep='\t')

    # aggregate by datetime & keyword
    hotlink_bbs = hotlink[hotlink.domain=='bbs'].pivot_table(index='datetime', columns='keyword', values='count', aggfunc=sum)
    hotlink_twitter = hotlink[hotlink.domain=='twitter_sampling'].pivot_table(index='datetime', columns='keyword', values='count', aggfunc=sum)
    hotlink_blog = hotlink[hotlink.domain=='blog'].pivot_table(index='datetime', columns='keyword', values='count', aggfunc=sum)

    # indexをdatetime型に変換
    hotlink_bbs.index = pd.to_datetime(hotlink_bbs.index)
    hotlink_twitter.index = pd.to_datetime(hotlink_twitter.index)
    hotlink_blog.index = pd.to_datetime(hotlink_blog.index)

    # 1日先へシフト
    hotlink_bbs = hotlink_bbs.shift()
    hotlink_twitter = hotlink_twitter.shift()
    hotlink_blog = hotlink_blog.shift()

    # カラム名を変更
    hotlink_bbs.columns = ['HOTLINK_BBS_'+ c for c in hotlink_bbs.columns]
    hotlink_twitter.columns = ['HOTLINK_TWITTER_'+ c for c in hotlink_twitter.columns]
    hotlink_blog.columns = ['HOTLINK_BLOG_'+ c for c in hotlink_blog.columns]

    # merge
    hotlink = pd.concat([hotlink_bbs, hotlink_twitter, hotlink_blog], axis=1)

    return hotlink

# Preprocess nied_oyama.tsv
def nied_oyama(num_rows=None):
    nied_oyama = pd.read_csv('../input/nied_oyama.tsv', sep='\t')

    # 日付を追加
    nied_oyama['datetime'] = pd.DatetimeIndex(pd.to_datetime(nied_oyama['日時'])).normalize()

    # 公園名を追加
    nied_oyama['park'] = '大山隠岐国立公園'

    # 日付・公園ごとに集計
    nied_oyama = nied_oyama.groupby(['datetime', 'park']).mean()

    # 1日先へシフト
    nied_oyama = nied_oyama.shift()

    # カラム名を変更
    nied_oyama.columns = ['NIED_OYAMA_'+ c for c in nied_oyama.columns]

    return nied_oyama

# Preprocess nightley.tsv
def nightley(num_rows=None):
    nightley = pd.read_csv('../input/nightley.tsv', sep='\t')
    nightley.loc[:,'datetime'] = pd.to_datetime(nightley['datetime'])
    nightley['park'] = '日光国立公園'

    # 1日先へシフト
    nightley[['Japan_count','Foreign_count']] = nightley[['Japan_count','Foreign_count']].shift()

    # additional feature
    nightley['NIGHTLEY_F_J_RATIO'] = nightley['Foreign_count'] / nightley['Japan_count']
    nightley['NIGHTLEY_F_J_SUM'] = nightley['Foreign_count'] + nightley['Japan_count']

    # 日付と公園毎に集計
    nightley = nightley.groupby(['datetime', 'park']).mean()

    # カラム名を変更
    nightley.columns = ['NIGHTLEY_'+ c for c in nightley.columns]

    return nightley

# Preprocess weather.tsv
def weather(num_rows=None):
    weather = pd.read_csv('../input/weather.tsv', sep='\t')
    weather['datetime'] = pd.to_datetime(weather['年月日'])

    # 公園と紐付け
    weather['park'] = weather['地点'].map(PARK_POINT)

    # 不要なカラムを削除
    feats = [c for c in weather.columns if c not in ['年月日', '地点', '天気概況(昼:06時~18時)', '天気概況(夜:18時~翌日06時)']]

    # 日付と公園ごとに集計
    weather = weather[feats].groupby(['park', 'datetime']).mean()

    # １日前にシフト
    for park in PARKS.keys():
        weather.loc[park, :] = weather[weather.loc[park, :]==park].shift()

    # カラム名を変更
    weather.columns = ['WEATHER_'+ c for c in weather.columns]

    return weather

# Preprocess jorudan.tsv
def jorudan(num_rows=None):
    jorudan = pd.read_csv('../input/jorudan.tsv', sep='\t')

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
    df = df.join(nightley(num_rows), how='left', on='datetime')

    # hotlink
    df = df.join(hotlink(num_rows), how='left', on='datetime')

    print(df)
