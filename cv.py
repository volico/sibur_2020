def get_indices(data, fold_times, first_train = True):
    cv = []
    for fold_time in fold_times:
        train = data[data['timestamp']< fold_time[0]].index
        train = list(train)
        test = data[(data['timestamp'] >= fold_time[0]) & (data['timestamp'] < fold_time[1])].index
        test = list(test)
        cv.append([train, test])

    if first_train == True:
        first_train = cv[0][0]
        t = 0
        for fold in cv:
            fold[0] = first_train
            t = t + 1

    return(cv)