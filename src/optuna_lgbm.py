
import lightgbm
import numpy as np
import pandas as pd
import optuna
import gc

from sklearn.model_selection import KFold, StratifiedKFold

from preprocess import train_test, nightley, hotlink, colopl, weather, nied_oyama, jorudan, agoop
from utils import FEATS_EXCLUDED, NUM_FOLDS, loadpkl

################################################################################
# optunaによるhyper parameter最適化
# 参考: https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py
################################################################################

NUM_ROWS=None
USE_PKL=True

if USE_PKL:
    DF = loadpkl('../output/df.pkl')
else:
    DF = train_test(NUM_ROWS)
    DF = pd.merge(DF, nightley(NUM_ROWS), on=['datetime', 'park'], how='outer')
    DF = pd.merge(DF, hotlink(NUM_ROWS), on='datetime', how='outer')
    DF = pd.merge(DF, colopl(NUM_ROWS), on=['year','month'], how='outer')
    DF = pd.merge(DF, weather(NUM_ROWS), on=['datetime', 'park'], how='outer')
    DF = pd.merge(DF, nied_oyama(NUM_ROWS), on=['datetime', 'park'], how='outer')
    DF = pd.merge(DF, agoop(num_rows), on=['park', 'year','month'], how='outer')
    DF = pd.merge(DF, jorudan(num_rows), on=['datetime', 'park'], how='outer')

# split test & train
TRAIN_DF = DF[DF['visitors'].notnull()]
FEATS = [f for f in TRAIN_DF.columns if f not in FEATS_EXCLUDED]

def objective(trial):
    lgbm_train = lightgbm.Dataset(TRAIN_DF[FEATS],
                                  np.log1p(TRAIN_DF['visitors']),
                                  free_raw_data=False
                                  )

#    num_round = trial.suggest_int('num_round', 1, 500)
    params = {'objective': 'regression',
              'metric': 'rmse',
              'verbosity': -1,
              "learning_rate": 0.01,
              'device': 'gpu',
              'seed': 326,
              'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
              'num_leaves': trial.suggest_int('num_leaves', 16, 64),
              'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.001, 1),
              'subsample': trial.suggest_uniform('subsample', 0.001, 1),
              'max_depth': trial.suggest_int('num_leaves', 16, 64),
              'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 10),
              'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 10),
              'min_split_gain': trial.suggest_uniform('min_split_gain', 0, 10),
              'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 45),
              'min_data_in_leaf': trial.suggest_int('num_leaves', 16, 64)
              }

    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if params['boosting_type'] == 'goss':
        params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate'])

    folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=47)

    clf = lightgbm.cv(params=params,
                      train_set=lgbm_train,
                      metrics=['rmse'],
                      nfold=NUM_FOLDS,
                      folds=folds.split(TRAIN_DF[FEATS], TRAIN_DF['park_japanese_holiday']),
                      num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                      early_stopping_rounds=200,
                      verbose_eval=100,
                      seed=47,
                     )
    gc.collect()
    return clf['rmse-mean'][-1]

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=20)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # save result
    hist_df = study.trials_dataframe()
    hist_df.to_csv("../output/optuna_result_lgbm.csv")
