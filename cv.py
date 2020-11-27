def get_indices(data, fold_times):
    cv = []
    for fold_time in fold_times:
        train = data[data['timestamp']< fold_time[0]].index
        train = list(train)
        test = data[(data['timestamp'] >= fold_time[0]) & (data['timestamp'] < fold_time[1])].index
        test = list(test)
        cv.append([train, test])

    return(cv)