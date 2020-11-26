def get_indices(data, fold_times):
    cv = []
    for fold_time in fold_times:
        train = data[data['timestamp']< fold_time[0]].index
        test = data[(data['timestamp'] >= fold_time[0]) & (data['timestamp'] < fold_time[1])].index
        cv.append([train, test])

    return(cv)