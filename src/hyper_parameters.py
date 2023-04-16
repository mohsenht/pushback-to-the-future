ARRIVAL_TO_GATE_WEIGHTED_MEAN_TIME_CONSTANT_FOR_EMPTY_ARRIVAL = 10

XGBOOST_PARAMETERS = {
    'objective': 'reg:squaredlogerror',
    'lambda': 0.8,
    'tree_method': 'hist',
    'max_bin': 24,
    'max_depth': 10,
    'eval_metric': 'mae'
}

XGBOOST_ESTIMATORS = 100
