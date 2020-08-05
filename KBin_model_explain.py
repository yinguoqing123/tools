from sklearn.preprocessing import KBinsDiscretizer
test_set_1['positive_p'] = model_xgb_1.predict_proba(test_X_1)[:, 1]
test_set_2['positive_p'] = model_xgb_2.predict_proba(test_X_2)[:, 1]
est1 = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit(test_set_2[['positive_p']])
est2 = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit(test_set_2[['positive_p']])
test_set_1['posibility_bin'] = est1.transform(test_set_1[['positive_p']]).reshape(-1)
test_set_2['posibility_bin'] = est2.transform(test_set_2[['positive_p']]).reshape(-1)
feature_mean_1 = test_set_1.groupby('posibility_bin')[numeric_list_1].mean()
