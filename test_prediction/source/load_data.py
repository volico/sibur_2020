import pandas as pd

def load(data_path, drop_start = True):
    train_features = pd.read_csv(data_path + 'train_features.csv')
    test_features = pd.read_csv(data_path + 'test_features.csv')
    train_targets = pd.read_csv(data_path + 'train_targets.csv')

    train_features['timestamp'] = pd.to_datetime(train_features['timestamp'])
    test_features['timestamp'] = pd.to_datetime(test_features['timestamp'])
    train_targets['timestamp'] = pd.to_datetime(train_targets['timestamp'])

    train_features = train_features.sort_values('timestamp').reset_index(drop=True)
    test_features = test_features.sort_values('timestamp').reset_index(drop=True)
    train_targets = train_targets.sort_values('timestamp').reset_index(drop=True)

    train_features = train_features.fillna(method='ffill')
    test_features = test_features.fillna(method='ffill')
    train_targets = train_targets.fillna(method='ffill')

    train_features = train_features.dropna().reset_index(drop=True)
    train_targets = train_targets[train_targets['timestamp'] >= pd.to_datetime('2020-01-01 04:30:00')].reset_index(
        drop=True)

    if drop_start == True:
        train_features = train_features[
            (train_features['timestamp'] > pd.to_datetime('2020-02-15 00:00:00'))].reset_index(drop=True)
        train_targets = train_targets[(train_targets['timestamp'] > pd.to_datetime('2020-02-15 00:00:00'))].reset_index(
            drop=True)

    return(train_features, train_targets, test_features)