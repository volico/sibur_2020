import pandas as pd
import numpy as np

def preprocessing(X, y, cv, params):
    print('current params:', params)
    n_in = params['n_in']
    params.pop('n_in')

    print('X shape before dropped nans', X.shape)
    n_backs = []
    for feature in [
        'A_CH4',
        'A_C2H6',
        'A_C3H8',
        'A_iC4H10',
        'A_nC4H10',
        'A_iC5H12',
        'A_nC5H12',
        'A_C6H14'
    ]:
        X[feature] = X[feature].shift(params['n_back_'+feature])
        n_backs.append(params['n_back_'+feature])
        params.pop('n_back_'+feature)
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

#    X = prepare_features(n_in, X)
    X = X.drop(['timestamp'], axis = 'columns')
    y = y[n_in + np.max(n_backs):]

    t = 0
    for fold in cv:
        fold[0] = fold[0][n_in + np.max(n_backs):]
        for idx1 in range(0, len(fold[0])):
            fold[0][idx1] = fold[0][idx1] - n_in - np.max(n_backs)

        for idx2 in range(0, len(fold[1])):
            fold[1][idx2] = fold[1][idx2] -  np.max(n_backs)
        cv[t] = fold
        t = t + 1
    print('cv after trans:', len(cv[0][0]))
    print('shape of X after trans:', X.shape)
    print('shape of y after trans:', y.shape)
    return X.values, y, cv, params