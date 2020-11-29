import pandas as pd
import numpy as np

def preprocessing(X, y, cv, params, pop = True):
    print('current params:', params)
    n_back = params['n_back']
    params.pop('n_back')
    n_in = params['n_in']
    params.pop('n_in')

    print('X shape before dropped nans', X.shape)
    X = X.shift(n_back)
    X = X.dropna()
    X = X.reset_index(drop=True)
    print('cv before trans:', len(cv[0][0]))
    print('Initial shape of X after dropping nans:', X.shape)
    print('Initial shape of y:', y.shape)

    def prepare_features(n_in, train_features):
        train_features['grouper'] = 1
        final_features = pd.DataFrame()
        for n in range(n_in, train_features.shape[0]):
            n_in_previous = train_features.iloc[(n - n_in):(n + 1), :]
            previous_pivoted = pd.pivot_table(n_in_previous, index='grouper', columns='timestamp')
            previous_pivoted.columns = previous_pivoted.columns.get_level_values(0) + \
                                       '_' + \
                                       pd.Series(list(range(0, n_in + 1)) * 10).astype(str)
            final_features = final_features.append(previous_pivoted.transpose()[1])
        final_features.index = train_features.iloc[n_in:, :].index

        return final_features

    X = prepare_features(n_in, X)
    y = y[n_in + n_back:]['B_C2H6']

    t = 0
    for fold in cv:
        fold[0] = fold[0][n_in + n_back:]
        for idx1 in range(0, len(fold[0])):
            fold[0][idx1] = fold[0][idx1] - n_in - n_back

        for idx2 in range(0, len(fold[1])):
            fold[1][idx2] = fold[1][idx2] - n_in - n_back
        cv[t] = fold
        t = t + 1
    print('cv after trans:', len(cv[0][0]))
    print('shape of X after trans:', X.shape)
    print('shape of y after trans:', y.shape)
    return X, y, cv, params