
import lightgbm as lgb
import numpy as np
import pandas as pd
import gc

from sklearn.metrics import mean_absolute_error

from utils import line_notify, loadpkl

################################################################################
# Preprocessingで作成したテストデータ及びLearningで作成したモデルを読み込み、予測結果をファイルとして出力するモジュール。
################################################################################

def main():
    # submitファイルをロード
    sub = pd.read_csv("../input/sample_submit.tsv",sep='\t', header=None)
    sub_lgbm = pd.read_csv("../output/submission_lgbm.tsv",sep='\t', header=None)
    sub_xgb = pd.read_csv("../output/submission_xgb.tsv",sep='\t', header=None)

    # カラム名を変更
    sub.columns =['index', 'visitors']
    sub_lgbm.columns =['index', 'visitors']
    sub_xgb.columns =['index', 'visitors']

    # merge
    sub.loc[:,'visitors'] = 0.5*sub_lgbm['visitors']+0.5*sub_xgb['visitors']

    del sub_lgbm, sub_xgb
    gc.collect()

    # out of foldの予測値をロード
    oof_lgbm = pd.read_csv("../output/oof_lgbm.csv")
    oof_xgb = pd.read_csv("../output/oof_xgb.csv")
    oof_preds = 0.5*oof_lgbm['OOF_PRED']+0.5*oof_xgb['OOF_PRED']

    # train_dfをロード
    train_df = loadpkl('../output/train_df.pkl')
    train_df = train_df.sort_values('index')

    # local cv scoreを算出
    local_mae = mean_absolute_error(train_df['visitors'], oof_preds)

    # LINE通知
    line_notify('Blend Local MAE score %.6f' % local_mae)

    del oof_lgbm, oof_xgb
    gc.collect()

    # save submit file
    sub[['index', 'visitors']].sort_values('index').to_csv(submission_file_name, index=False, header=False, sep='\t')

if __name__ == '__main__':
    submission_file_name = "../output/submission_blend.csv"
    main()
