import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import s3fs
from sklearn.metrics import precision_recall_curve
import lightgbm as lgb
import joblib
from bayes_opt import BayesianOptimization
import numpy as np

def lgb_eval(num_leaves,  max_depth, lambda_l2,lambda_l1, min_child_samples, bagging_fraction,
             feature_fraction, min_child_weight):
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "lambda_l2": lambda_l2,
        "lambda_l1": lambda_l1,
        "num_threads": 32,
        "min_child_samples": int(min_child_samples),
        "min_child_weight": min_child_weight,
        "learning_rate": 0.05,
        "bagging_fraction": bagging_fraction,
        "feature_fraction": feature_fraction,
        "seed": 2020,
        "verbosity": -1
    }
    train_df = lgb.Dataset(train_X, train_set.label)
    scores = lgb.cv(params, train_df, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=False,
                     nfold=3)['auc-mean'][-1]
    return scores

def param_tuning(init_points,num_iter,**args):
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (10, 120),
                                                'max_depth': (5, 15),
                                                'lambda_l2': (0.0, 3),
                                                'lambda_l1': (0.0, 3),
                                                'bagging_fraction': (0.5, 0.8),
                                                'feature_fraction': (0.5, 0.8),
                                                'min_child_samples': (20, 100),
                                                'min_child_weight': (0, 15)
                                                })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
    return lgbBO
    
  result = param_tuning(5, 40)
  print('模型最佳参数AUC:{}'.format(result.max['target']))
  params = result.max['params']
  print('模型最佳参数:', '\n', params)
  result.probe(params)
  result.maximize(0, 10)
  params = result.max['params']
  print('再次精调后最佳AUC:{}'.format(result.max['target']))
  print('再次精调后最佳参数：', '\n', result.max['params'])
  model_lgb = lgb.LGBMClassifier(n_estimators=1000, max_depth=int(params['max_depth']),
                                 num_leaves=int(params['num_leaves']), n_jobs=-1, learning_rate=0.05,
                                 colsample_bytree=params['bagging_fraction'], subsample=params['feature_fraction'],
                                 reg_lambda=params['lambda_l1'], reg_alpha=params['lambda_l2'],
                                 min_child_weight=params['min_child_weight'],
                                 min_child_samples=int(params['min_child_samples']),
                             random_state=2019)
  model_lgb.fit(train_X, train_set.label, eval_set=[(test_X, test_set.label)], eval_metric='auc',
                early_stopping_rounds=30)
