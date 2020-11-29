import pandas as pd
import numpy as np

def preprocessing(X, y, cv, params, pop = True):
    print('current params:', params)
    seq_len = params['seq_len']
    n_back = params['n_back']
    n_rolling = params['n_rolling']
    if pop == True:
        params.pop('seq_len')
        params.pop('n_back')
        params.pop('n_rolling')

    print('Initial shape of X:', X.shape)
    cols = X.columns
    X_rolling = X.rolling(n_rolling).mean()
    X_rolling.columns = [col + '_rolling' for col in cols]
    X_shifted = X.shift(n_back)

    X = pd.concat([X_rolling, X_shifted], axis='columns')

    X = X.dropna()

    print('cv before trans:', len(cv[0][0]))
    print('Initial shape of X after rolling:', X.shape)
    print('Initial shape of y:', y.shape)

    def make_seqs(n, data):
        final_X = []

        def seq2seq(x):
            seqs_X.append(np.array(x).reshape(-1, 1))
            return (5)

        for col in data.columns:
            seqs_X = []
            data[col].rolling(n).apply(seq2seq)
            seqs_X = np.array(seqs_X)
            final_X.append(seqs_X)

        final_X = np.concatenate(final_X, axis=2)
        return (final_X)

    X = make_seqs(seq_len, X)
    y = y.iloc[((seq_len - 1) + max(n_rolling - 1, n_back)):]

    t = 0
    for fold in cv:
        fold[0] = fold[0][(seq_len - 1) + max(n_rolling - 1, n_back):]
        for idx1 in range(0, len(fold[0])):
            fold[0][idx1] = fold[0][idx1] - (seq_len - 1) - max(n_rolling - 1, n_back)

        for idx2 in range(0, len(fold[1])):
            fold[1][idx2] = fold[1][idx2] - (seq_len - 1) - max(n_rolling - 1, n_back)
        cv[t] = fold
        t = t + 1
    print('cv after trans:', len(cv[0][0]))
    print('shape of X after trans:', X.shape)
    print('shape of y after trans:', y.shape)
    return X, y, cv, params