#!/bin/sh
python train_lgbm.py
python train_xgb.py
python predict.py
