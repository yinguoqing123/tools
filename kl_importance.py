def cal_kl(data_set, key):
    # index中含有nan时，reset_index会出错
    if data_set[key].nunique()<=10:
        tmp = data_set.groupby('label')[key].value_counts(normalize=True, dropna=False).to_frame()\
            .add_suffix('_count').reset_index(level=0)
        tmp = tmp.pivot(columns='label', values='{}_count'.format(key))
        print(tmp)
        return tmp.apply(lambda x: -x[0]*np.log2(x[1]/x[0]), axis=1).sum()
    else:
        data_set[key] = pd.qcut(data_set[key], [0.1*i for i in range(11)], duplicates='drop')
        tmp = data_set.groupby('label')[key].value_counts(normalize=True, dropna=False).to_frame() \
            .add_suffix('_count').reset_index(level=0)
        tmp = tmp.pivot(columns='label', values='{}_count'.format(key))
        print(tmp)
        return tmp.apply(lambda x: -x[0]*np.log2(x[1]/x[0]), axis=1).sum()
        
 
