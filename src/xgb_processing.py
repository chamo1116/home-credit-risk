import xgboost as xgb
import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score

def xgb_validate(x_trn, x_val, y_trn, y_val, xgb_params, seed_val=0, num_rounds=4096):
	num_rounds = num_rounds
	xgtrain = xgb.DMatrix(x_trn, label=y_trn)
	xgtest = xgb.DMatrix(x_val, label=y_val)

	# train
	watchlist = [ (xgtest, 'test') ]
	model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

	# predict
	y_pred = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

	# eval
	val_score = roc_auc_score(y_val, y_pred)
	return val_score
def xgb_cross_val(params, X, y, folds):
	n = 1
	num_rounds = 3000

	list_rounds = []
	list_scores = []

	for train_idx, valid_idx in folds:
		xgtrain = xgb.DMatrix(X.values[train_idx], label=y.values[train_idx])
		xgtest = xgb.DMatrix(X.values[valid_idx], label=y.values[valid_idx])

		watchlist = [ (xgtest, 'test') ]
		model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=False)

		rounds = model.best_ntree_limit
		score = model.best_score

		list_rounds.append(rounds)
		list_scores.append(score)
		n += 1

	mean_round = np.mean(list_rounds)

	return mean_round

def xgb_features_importance(classifier, feat_names):
	importance = classifier.get_fscore()
	result = pd.DataFrame(importance,index=np.arange(2)).T
	result.iloc[:,0]= result.index
	result.columns=['feature','importance']

	result_by_importance = result.sort_values('importance', inplace=False, ascending=False)
	result_by_importance.reset_index(drop=True, inplace=True)

	result_by_feature = result.sort_values('feature', inplace=False, ascending=True)
	result_by_feature.reset_index(drop=True, inplace=True)

	return result_by_importance, result_by_feature

def xgb_output(X_test, sk_id_curr, classifier, n_stop, val_score):
	# XGBoost
	xgtest = xgb.DMatrix(X_test)
	predictions = classifier.predict(xgtest, ntree_limit=n_stop)

	result = pd.DataFrame({
	'SK_ID_CURR': sk_id_curr,
	'TARGET': predictions
	})


	# Features importance
	fi_by_importance, fi_by_feature = xgb_features_importance(classifier, X_test.columns)
	
	print('\n\nHere are the top 40 important features')
	print(fi_by_importance.head(40))