#!/bin/sh
python BayesOptLGBM.py
python BayesOptXGB.py
python optuna_lgbm.py
python optuna_xgb.py
