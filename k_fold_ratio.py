def k_flod_ratio_feature(data_set, key, k, flag):
    if flag=='train':
        key_ratio = []
        for i in range(k):
            tmp = data_set[data_set.k_fold!=i]
            tmp_key = tmp.groupby(key, as_index=False).label.agg({'{}_ratio'.format(key): 'mean',
                                                                  '{}_sum_orders'.format(key): 'count',
                                                                  '{}_neg_orders'.format(key): 'sum'}
                                                                 )
            tmp_key['k_fold'] = i
            key_ratio.append(tmp_key)
        return pd.concat(key_ratio)
    else:
        key_ratio = data_set.groupby(key, as_index=False).label.agg({'{}_ratio'.format(key): 'mean',
                                                              '{}_sum_orders'.format(key): 'count',
                                                              '{}_neg_orders'.format(key): 'sum'})
        key_ratio['{}_sum_orders'.format(key)] = (k-1)/k*key_ratio['{}_sum_orders'.format(key)]
        key_ratio['{}_neg_orders'.format(key)] = (k - 1) / k * key_ratio['{}_neg_orders'.format(key)]
        return key_ratio


train_set.loc[:, 'k_fold'] = 0
train_set.k_fold = train_set.k_fold.apply(lambda x:np.random.randint(5))
country_ratio = k_flod_ratio_feature(train_set.copy(), 'shpp_country_nm', 5, 'train')
prvn_ratio = k_flod_ratio_feature(train_set.copy(), 'prvn_nm', 5, 'train')
city_ratio = k_flod_ratio_feature(train_set.copy(), 'city_nm', 5, 'train')
train_set = train_set.merge(country_ratio, on=['shpp_country_nm', 'k_fold'], how='left')
train_set = train_set.merge(prvn_ratio, on=['prvn_nm', 'k_fold'], how='left')
train_set = train_set.merge(city_ratio, on=['city_nm', 'k_fold'], how='left')

country_ratio_test = k_flod_ratio_feature(train_set.copy(), 'shpp_country_nm', 5, 'test')
prvn_ratio_test = k_flod_ratio_feature(train_set.copy(), 'prvn_nm', 5, 'test')
city_ratio_test = k_flod_ratio_feature(train_set.copy(), 'city_nm', 5, 'test')
test_set = test_set.merge(country_ratio_test, on=['shpp_country_nm'], how='left')
test_set = test_set.merge(prvn_ratio_test, on=['prvn_nm'], how='left')
test_set = test_set.merge(city_ratio_test, on=['city_nm'], how='left')
