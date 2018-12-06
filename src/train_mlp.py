
import gc
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
import time
import warnings

from contextlib import contextmanager
from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Model
from keras.optimizers import Adam
from matplotlib.font_manager import FontProperties
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold

from preprocess import train_test, nightley, hotlink, colopl, weather, nied_oyama, jorudan, agoop
from utils import line_notify, NUM_FOLDS, FEATS_EXCLUDED, loadpkl, save2pkl, PARKS

# 日本語表示用の設定
font_path = '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

################################################################################
# Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
# 学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
################################################################################

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# メルカリコンペのwinning solutionをコピペ
# https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
def model_mlp(input_dim):
    model_in = Input(shape=(input_dim,), dtype='float32', sparse=True)
    out = Dense(192, activation='relu')(model_in)
    out = Dense(64, activation='relu')(out)
    out = Dense(64, activation='relu')(out)
    out = Dense(1)(out)
    return Model(model_in, out)

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['visitors'].notnull()]
    test_df = df[df['visitors'].isnull()]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # save pkl
    save2pkl('../output/train_df.pkl', train_df)
    save2pkl('../output/test_df.pkl', test_df)

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['park'])):
        train_x, train_y = train_df[feats].iloc[train_idx], np.log1p(train_df['visitors'].iloc[train_idx])
        valid_x, valid_y = train_df[feats].iloc[valid_idx], np.log1p(train_df['visitors'].iloc[valid_idx])

        # model
        reg = model_mlp(train_x.shape[1])

        # compile
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=3e-3)
                      )

        # early stopping
        early_stopping = EarlyStopping(monitor='val_my_iou_metric',
                                       mode='max',
                                       patience=15,
                                       verbose=1)

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save model
        reg.save_model('../output/lgbm_'+str(n_fold)+'.txt')

        oof_preds[valid_idx] = np.expm1(reg.predict(valid_x, num_iteration=reg.best_iteration))
        sub_preds += np.expm1(reg.predict(test_df[feats], num_iteration=reg.best_iteration)) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d MAE : %.6f' % (n_fold + 1, mean_absolute_error(np.expm1(valid_y), oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full MAEスコアの表示&LINE通知
    full_mae = mean_absolute_error(train_df['visitors'], oof_preds)
    line_notify('Full MAE score %.6f' % full_mae)

    if not debug:
        # 提出データの予測値を保存
        test_df.loc[:,'visitors'] = sub_preds
        test_df[['index', 'visitors']].sort_values('index').to_csv(submission_file_name, index=False, header=False, sep='\t')

        # out of foldの予測値を保存
        train_df.loc[:,'OOF_PRED'] = oof_preds
        train_df[['index', 'OOF_PRED']].sort_values('index').to_csv(oof_file_name, index= False)

    return feature_importance_df

def main(debug=False, use_pkl=False):
    num_rows = 10000 if debug else None
    if use_pkl:
        df = loadpkl('../output/df.pkl')
    else:
        with timer("train & test"):
            df = train_test(num_rows)
        with timer("nightley"):
            df = pd.merge(df, nightley(num_rows), on=['datetime', 'park'], how='outer')
        with timer("hotlink"):
            df = pd.merge(df, hotlink(num_rows), on='datetime', how='outer')
        with timer("colopl"):
            df = pd.merge(df, colopl(num_rows), on=['park', 'year', 'month'], how='outer')
#        with timer("weather"):
#            df = pd.merge(df, weather(num_rows), on=['datetime', 'park'], how='outer')
        with timer("nied_oyama"):
            df = pd.merge(df, nied_oyama(num_rows), on=['datetime', 'park'], how='outer')
        with timer("agoop"):
            df = pd.merge(df, agoop(num_rows), on=['park', 'year','month'], how='outer')
        with timer("jorudan"):
            df = pd.merge(df, jorudan(num_rows), on=['datetime', 'park'], how='outer')
        with timer("save pkl"):
            save2pkl('../output/df.pkl', df)
    with timer("Run LightGBM with kfold"):
        print("df shape:", df.shape)
        feat_importance = kfold_lightgbm(df, num_folds=NUM_FOLDS, stratified=True, debug=debug)
        display_importances(feat_importance ,'../output/lgbm_importances.png', '../output/feature_importance_lgbm.csv')

if __name__ == "__main__":
    submission_file_name = "../output/submission.tsv"
    oof_file_name = "../output/oof_lgbm.csv"
    with timer("Full model run"):
        main(debug=False,use_pkl=False)
