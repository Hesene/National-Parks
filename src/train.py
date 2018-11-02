import gc
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold

from preprocess import train_test, nightley, hotlink
from utils import line_notify, NUM_FOLDS

# 日本語表示用の設定
font_path = '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

################################################################################
# Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
# 学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
################################################################################

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Display/plot feature importance
def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # importance下位の確認用に追加しました
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['visitors'].notnull()]
    test_df = df[df['visitors'].isnull()]

    # 確認用にcsvをsave
    train_df.to_csv('../output/train_df.csv')
    test_df.to_csv('../output/test_df.csv')

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['index', 'datetime', 'visitors', 'year', 'park', 'weekofyear']]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['weekofyear'])):
        train_x, train_y = train_df[feats].iloc[train_idx], np.log1p(train_df['visitors'].iloc[train_idx])
        valid_x, valid_y = train_df[feats].iloc[valid_idx], np.log1p(train_df['visitors'].iloc[valid_idx])

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # パラメータは適当です
        params ={
                'device' : 'gpu',
#                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'num_iteration': 10000,
                'learning_rate': 0.02,
                'num_leaves': 31,
                'colsample_bytree': 0.9,
                'subsample': 0.9,
#                'max_depth': 10,
#                'reg_alpha': 8.7511002653,
#                'reg_lambda': 2.2602432486,
#                'min_split_gain': 0.0503376564,
#                'min_child_weight': 45,
#                'min_data_in_leaf': 23,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
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
        test_df[['index', 'visitors']].to_csv(submission_file_name, index=False, header=False, sep='\t')

        # out of foldの予測値を保存
        train_df.loc[:,'OOF_PRED'] = oof_preds
        train_df[['index', 'OOF_PRED']].to_csv(oof_file_name, index= False)

    return feature_importance_df

def main(debug = False):
    num_rows = 10000 if debug else None
    with timer("train & test"):
        df = train_test(num_rows)
    with timer("nightley"):
        df = pd.merge(df, nightley(num_rows), on='datetime', how='outer')
    with timer("hotlink"):
        df = pd.merge(df, hotlink(num_rows), on='datetime', how='outer')
        print("df shape:", df.shape)
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds=NUM_FOLDS, stratified=True, debug=debug)
        display_importances(feat_importance ,'../output/lgbm_importances.png', '../output/feature_importance_lgbm.csv')

if __name__ == "__main__":
    submission_file_name = "../output/submission.tsv"
    oof_file_name = "../output/oof_lgbm.csv"
    with timer("Full model run"):
        main(debug=False)
