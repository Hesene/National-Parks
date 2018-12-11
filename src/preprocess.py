
import pandas as pd
import numpy as np
import gc
import os
import datetime

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

# GWのフラグ
def getGoldenWeek(date):
    gw  = ((date.dt.day==29)&(date.dt.month==4)).astype(int)
    gw += ((date.dt.day==30)&(date.dt.month==4)).astype(int)
    gw += ((date.dt.day==1)&(date.dt.month==5)).astype(int)
    gw += ((date.dt.day==2)&(date.dt.month==5)).astype(int)
    gw += ((date.dt.day==3)&(date.dt.month==5)).astype(int)
    gw += ((date.dt.day==4)&(date.dt.month==5)).astype(int)
    gw += ((date.dt.day==5)&(date.dt.month==5)).astype(int)
    return gw

# 年末年始のフラグ
def getNewYearsDay(date):
    nyd  = ((date.dt.day==30)&(date.dt.month==12)).astype(int)
    nyd += ((date.dt.day==31)&(date.dt.month==12)).astype(int)
    nyd += ((date.dt.day==1)&(date.dt.month==1)).astype(int)
    nyd += ((date.dt.day==2)&(date.dt.month==1)).astype(int)
    nyd += ((date.dt.day==3)&(date.dt.month==1)).astype(int)
    return nyd

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
    df['japanese_holiday'] = getJapaneseHolidays(df['datetime']).replace(2,1)

    # 連休数のファクターを生成
    holidays = df.groupby('datetime')['japanese_holiday'].mean().replace(2,1)
    holidays = fillHolidays(holidays).replace(2,1) # 休日の谷間の平日を休日にする
    df['num_holidays'] = df['datetime'].map(getNumHolidays(holidays))

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
    df['new_years_day'] = getNewYearsDay(df['datetime'])
    df['golden_week'] = getGoldenWeek(df['datetime'])

    df['park_day'] = df['park'].astype(str)+'_'+df['day'].astype(str)
    df['park_month'] = df['park'].astype(str)+'_'+df['month'].astype(str)
    df['park_weekday'] = df['park'].astype(str)+'_'+df['weekday'].astype(str)
    df['park_japanese_holiday'] = df['park'].astype(str)+'_'+df['japanese_holiday'].astype(str)
    df['park_weekofyear'] = df['park'].astype(str)+'_'+df['weekofyear'].astype(str)
    df['park_num_holiday'] = df['park'].astype(str)+'_'+df['num_holidays'].astype(str)
    df['park_new_years_day'] = df['park'].astype(str)+'_'+df['new_years_day'].astype(str)
    df['park_golden_week'] = df['park'].astype(str)+'_'+df['golden_week'].astype(str)

    # categorical変数を変換
    df_res, cat_cols = one_hot_encoder(df, nan_as_category=False)

    # stratify & mearge用
    df_res['park'] = df['park']
    df_res['weekofyear'] = df['weekofyear'].astype(int)
    df_res['weekday'] = df['weekday'].astype(int)
    df_res['year'] = df['datetime'].dt.year.astype(int)
    df_res['month'] = df['datetime'].dt.month.astype(int)
    df_res['park_month'], _ = pd.factorize(df['park_month'])
    df_res['park_japanese_holiday'], _ = pd.factorize(df['park_japanese_holiday'])
    df_res['ISESHIMA_summit'] = ((df['park']=='伊勢志摩国立公園')&df['japanese_holiday']&('2016-5-27'>df['datetime'])&(df['datetime']>'2015-6-5')).astype(int) # 2016年伊勢島サミット開催決定後の休日フラグ

    return df_res

# Preprocess colopl.tsv
def colopl(num_rows=None):
    colopl = pd.read_csv('../input/colopl.tsv', sep='\t')

    # 1-9をとりあえず5で埋めます
    colopl['count'] = colopl['count'].replace('1-9', 5).astype(int)

    # 月ごとに集計
    colopl = colopl.pivot_table(index=['park', 'year', 'month'],
                                columns='country_jp',
                                values='count',
                                aggfunc=[np.sum, 'mean', np.max])

    # nanを0埋め
    colopl.fillna(0, inplace=True)

    # 前月との差分データを追加
#    colopl_diff = colopl.diff()

    # カラム名を変更
    colopl.columns = pd.Index([e[1] + "_" + e[0].upper() for e in colopl.columns.tolist()])
    colopl.columns = ['COLOPL_'+ c for c in colopl.columns]
#    colopl_diff.columns = ['COLOPL_DIFF_'+ c for c in colopl_diff.columns]

    # merge
#    colopl = pd.concat([colopl, colopl_diff], axis=1)

    # indexをreset
    colopl=colopl.reset_index()

    #　１ヶ月先へシフト
    for i, (y, m) in enumerate(zip(colopl['year'], colopl['month'])):
        if m==12:
            colopl.loc[i,'month']-=11
            colopl.loc[i,'year']+=1
        else:
            colopl.loc[i,'month']+=1

    # 2018/1/1以降のデータを削除
    colopl = colopl[colopl['year']<2018]

#    del colopl_diff
    gc.collect()

    return colopl

# Preprocess hotlink.tsv
def hotlink(num_rows=None):
    # load csv
    hotlink = pd.read_csv('../input/hotlink.tsv', sep='\t')

    # aggregate by datetime & keyword
    hotlink_all = hotlink.pivot_table(index='datetime',columns='keyword', values='count', aggfunc=[np.sum, np.max, 'mean'])
    hotlink_bbs = hotlink[hotlink.domain=='bbs'].pivot_table(index='datetime', columns='keyword', values='count', aggfunc=[np.sum, np.max, 'mean'])
    hotlink_twitter = hotlink[hotlink.domain=='twitter_sampling'].pivot_table(index='datetime', columns='keyword', values='count', aggfunc=[np.sum, np.max, 'mean'])
    hotlink_blog = hotlink[hotlink.domain=='blog'].pivot_table(index='datetime', columns='keyword', values='count', aggfunc=[np.sum, np.max, 'mean'])

    # 欠損値をゼロ埋め
    hotlink_all.fillna(0, inplace=True)
    hotlink_bbs.fillna(0, inplace=True)
    hotlink_twitter.fillna(0, inplace=True)
    hotlink_blog.fillna(0, inplace=True)

    # indexをdatetime型に変換
    hotlink_all.index = pd.to_datetime(hotlink_all.index)
    hotlink_bbs.index = pd.to_datetime(hotlink_bbs.index)
    hotlink_twitter.index = pd.to_datetime(hotlink_twitter.index)
    hotlink_blog.index = pd.to_datetime(hotlink_blog.index)

    # 1日先へシフト
    hotlink_all = hotlink_all.shift()
    hotlink_bbs = hotlink_bbs.shift()
    hotlink_twitter = hotlink_twitter.shift()
    hotlink_blog = hotlink_blog.shift()

    # カラム名を変更
    hotlink_all.columns = pd.Index([e[1] + "_" + e[0].upper() for e in hotlink_all.columns.tolist()])
    hotlink_bbs.columns = pd.Index([e[1] + "_" + e[0].upper() for e in hotlink_bbs.columns.tolist()])
    hotlink_twitter.columns = pd.Index([e[1] + "_" + e[0].upper() for e in hotlink_twitter.columns.tolist()])
    hotlink_blog.columns = pd.Index([e[1] + "_" + e[0].upper() for e in hotlink_blog.columns.tolist()])

    hotlink_all.columns = ['HOTLINK_ALL_'+ c for c in hotlink_all.columns]
    hotlink_bbs.columns = ['HOTLINK_BBS_'+ c for c in hotlink_bbs.columns]
    hotlink_twitter.columns = ['HOTLINK_TWITTER_'+ c for c in hotlink_twitter.columns]
    hotlink_blog.columns = ['HOTLINK_BLOG_'+ c for c in hotlink_blog.columns]

    # merge
    hotlink = pd.concat([hotlink_all, hotlink_bbs, hotlink_twitter, hotlink_blog], axis=1)

    del hotlink_all, hotlink_bbs, hotlink_twitter, hotlink_blog
    gc.collect()

    return hotlink

# Preprocess nied_oyama.tsv
def nied_oyama(num_rows=None):
    nied_oyama = pd.read_csv('../input/nied_oyama.tsv', sep='\t')

    # 日付を追加
    nied_oyama['datetime'] = pd.DatetimeIndex(pd.to_datetime(nied_oyama['日時'])).normalize()

    # 公園名を追加
    nied_oyama['park'] = '大山隠岐国立公園'

    feats_nied_oyama = [c for c in nied_oyama.columns if c not in ['park', 'datetime', '日時']]

    # 集約用のdictを生成
    agg_nied_oyama = {}
    for c in feats_nied_oyama:
        agg_nied_oyama[c]=['max', 'min', 'mean', 'std']

    # 日付・公園ごとに集計
    nied_oyama = nied_oyama.groupby(['datetime', 'park']).agg(agg_nied_oyama)

    # ゼロ埋め
    nied_oyama.fillna(0, inplace=True)

    # 1日先へシフト
    nied_oyama = nied_oyama.shift()

    # カラム名を変更
    nied_oyama.columns = pd.Index([e[0] + "_" + e[1].upper() for e in nied_oyama.columns.tolist()])
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

    feats_nightley = [c for c in nightley.columns if c not in ['park', 'datetime']]

    # 集約用のdictを生成
    agg_nightley = {}
    for c in feats_nightley:
        agg_nightley[c]=['sum', 'mean', 'max', 'min', 'std']

    # 日付と公園毎に集計
    nightley = nightley.groupby(['datetime', 'park']).agg(agg_nightley)

    # ゼロ埋め
    nightley.fillna(0, inplace=True)

    # カラム名を変更
    nightley.columns = pd.Index([e[0] + "_" + e[1].upper() for e in nightley.columns.tolist()])
    nightley.columns = ['NIGHTLEY_'+ c for c in nightley.columns]

    return nightley

# Preprocess weather.tsv
def weather(num_rows=None):
    weather = pd.read_csv('../input/weather.tsv', sep='\t')
    weather['datetime'] = pd.to_datetime(weather['年月日'])

    # １日前にシフト
    weather['datetime'] = weather['datetime']+datetime.timedelta(1)

    # 公園と紐付け
    weather['park'] = weather['地点'].map(PARK_POINT)

    # 不要なカラムを削除
    feats_weather = [c for c in weather.columns if c not in ['年月日', '地点', '天気概況(昼:06時~18時)', '天気概況(夜:18時~翌日06時)', '最多風向(16方位)']]
    weather = weather[feats_weather]

    # 集約用のdictを生成
    agg_weather = {}
    for c in feats_weather:
        if c not in ['park', 'datetime']:
            agg_weather[c]=['min', 'max', 'mean', 'std']

    # 日付と公園ごとに集計
    weather = weather[feats_weather].groupby(['park', 'datetime']).agg(agg_weather)

    # カラム名を変更
    weather.columns = pd.Index([e[0] + "_" + e[1].upper() for e in weather.columns.tolist()])
    weather.columns = ['WEATHER_'+ c for c in weather.columns]

    return weather

# Preprocess jorudan.tsv
def jorudan(num_rows=None):
    tmp_jorudan = pd.read_csv('../input/jorudan.tsv', sep='\t', nrows=num_rows)

    # 日付をdatetime型へ変換
    tmp_jorudan['access_date'] = pd.to_datetime(tmp_jorudan['access_date'])
    tmp_jorudan['datetime'] = pd.to_datetime(tmp_jorudan['departure_and_arrival_date'])

    # 当日以降のアクセスデータを削除
    tmp_jorudan = tmp_jorudan[tmp_jorudan['datetime']>tmp_jorudan['access_date']]

    # 2018/1/1以降のデータを削除
    tmp_jorudan = tmp_jorudan[tmp_jorudan['datetime']<'2018-01-01']

    # one-hot encoding
    jorudan, cols = one_hot_encoder(tmp_jorudan[['departure_and_arrival_type',
                                                 'departure_and_arrival_place_type',
                                                 'departure_prefecture',
                                                 'arrival_prefecture']],
                                                 nan_as_category=False)

    # 日付と公園名のカラムを追加
    jorudan['park']=tmp_jorudan['park']
    jorudan['datetime']=tmp_jorudan['datetime']

    feats_jorudan = [c for c in jorudan.columns if c not in ['park', 'datetime']]

    # 集約用のdictを生成
    agg_jorudan = {}
    for c in feats_jorudan:
        agg_jorudan[c]=['sum', 'mean']

    # 日付と公園名で集約
    jorudan = jorudan.groupby(['park', 'datetime']).agg(agg_jorudan)

    # ゼロ埋め
    jorudan.fillna(0, inplace=True)

    # カラム名の変更
    jorudan.columns = pd.Index([e[0] + "_" + e[1].upper() for e in jorudan.columns.tolist()])

    # 追加の特徴量
    jorudan['departure_and_arrival_place_mean_sum'] = jorudan['departure_and_arrival_place_type_A_MEAN']+jorudan['departure_and_arrival_place_type_D_MEAN']
    jorudan['departure_and_arrival_place_sum_sum'] = jorudan['departure_and_arrival_place_type_A_SUM']+jorudan['departure_and_arrival_place_type_D_SUM']
    jorudan['departure_and_arrival_type__mean_sum'] = jorudan['departure_and_arrival_type_A_MEAN']+jorudan['departure_and_arrival_type_D_MEAN']
    jorudan['departure_and_arrival_type_sum_sum'] = jorudan['departure_and_arrival_type_A_SUM']+jorudan['departure_and_arrival_type_D_SUM']
    jorudan['departure_and_arrival_place_mean_ratio'] = jorudan['departure_and_arrival_place_type_A_MEAN']/jorudan['departure_and_arrival_place_type_D_MEAN']
    jorudan['departure_and_arrival_place_sum_ratio'] = jorudan['departure_and_arrival_place_type_A_SUM']/jorudan['departure_and_arrival_place_type_D_SUM']
    jorudan['departure_and_arrival_type_mean_ratio'] = jorudan['departure_and_arrival_type_A_MEAN']/jorudan['departure_and_arrival_type_D_MEAN']
    jorudan['departure_and_arrival_type_sum_ratio'] = jorudan['departure_and_arrival_type_A_SUM']/jorudan['departure_and_arrival_type_D_SUM']

    # カラム名を変更
    jorudan.columns = ['JORUDAN_'+ c for c in jorudan.columns]

    del tmp_jorudan
    gc.collect()

    return jorudan

# Preprocess agoop.tsv
def agoop(num_rows=None):

    agoop =pd.DataFrame()

    for filename in os.listdir('../input/agoop/'):
        if 'month_time_mesh100m_' in filename:
            # load tsv
            tmp_agoop = pd.read_csv('../input/agoop/'+filename, sep='\t')

            # pivot tableで集約
            tmp_agoop = tmp_agoop.pivot_table(index=['park', 'year', 'month'],
                                              columns=['dayflag', 'hour'],
                                              values='population',
                                              aggfunc=[np.sum, 'mean', 'max', 'min', 'std'])

            # カラム名を変更
            tmp_agoop.columns = ['AGOOP_dayflag'+str(tup[1])+'_'+'hour'+str(tup[2])+'_'+tup[0].upper() for tup in tmp_agoop.columns.values]

            # merge
            agoop = agoop.append(tmp_agoop)

            del tmp_agoop
            gc.collect()

            print(filename+' done.')

    agoop = agoop.reset_index()

    # １ヶ月先にシフト
    for i, (y, m) in enumerate(zip(agoop['year'], agoop['month'])):
        if m==12:
            agoop.loc[i,'month']-=11
            agoop.loc[i,'year']+=1
        else:
            agoop.loc[i,'month']+=1

    # 2018/1/1以降のデータを削除
    agoop = agoop[agoop['year']<2018]

    return agoop

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
