# 减小内存 
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage(deep='True').sum() / 1024**2
    df_int = df.select_dtypes(include=['int']) 
    convert_int = df_int.apply(pd.to_numeric, downcast='integer')
    df_float = df.select_dtypes(include=['float'])
    convert_float = df_float.apply(pd.to_numeric, downcast='float')
    df[convert_int.columns] = convert_int
    df[convert_float.columns] = convert_float
    end_mem = df.memory_usage(deep='True').sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# feature selection 
https://github.com/duxuhao/Feature-Selection
