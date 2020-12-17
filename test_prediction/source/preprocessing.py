import numpy as np

def preprocessing(X, y, cv, params):
    '''
    X - фичи (pandas.DataFrame)
    y - таргеты (pandas.DataFrame)
    params - параметры модели (dict)
    '''
    print('current params:', params)
    target = params['target']
    params.pop('target')
    print(target)

    print('X shape before dropped nans', X.shape)
    n_backs = []
    for feature in [

        'A_{}'.format(target)
    ]:
        X[feature] = X[feature].shift(params['n_back_'+feature])
        n_backs.append(params['n_back_'+feature])
        params.pop('n_back_'+feature)
    X = X.dropna()
    X = X.reset_index(drop=True)
    X = X[['A_' + target]]
    print('cv before trans:', len(cv[0][0]))
    print('Initial shape of X after dropping nans:', X.shape)
    print('Initial shape of y:', y.shape)

    y = y[np.max(n_backs):]['B_' + target]

    t = 0
    for fold in cv:
        fold[0] = fold[0][np.max(n_backs):]
        for idx1 in range(0, len(fold[0])):
            fold[0][idx1] = fold[0][idx1]- np.max(n_backs)

        for idx2 in range(0, len(fold[1])):
            fold[1][idx2] = fold[1][idx2] -  np.max(n_backs)
        cv[t] = fold
        t = t + 1
    print('cv after trans:', len(cv[0][0]))
    print('shape of X after trans:', X.shape)
    print('shape of y after trans:', y.shape)
    return X.values, y.values, cv, params