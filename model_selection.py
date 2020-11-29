import pandas as pd
import neptune
import optuna
import neptunecontrib.monitoring.optuna as optuna_utils
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

class training:

    def __init__(self, nn_model = None, training_nn = None,
                 name=None, description=None, params = None, properties=None, tags=None, upload_source_files=None):

        ## Создаем эксперимент в neptune
        neptune.create_experiment(name, description, params, properties, tags, upload_source_files)
        self.nn_model = nn_model
        self.training_nn = training_nn

    def set_up_studying(self, random_state, direction='minimize'):

        ## Выбираем стандартный сэмплер для подборки параметров
        sampler = optuna.samplers.TPESampler(seed=random_state)

        self.study = optuna.create_study(sampler=sampler, direction='minimize')

    def train(self, X, y = None, cv=None, model=None, params_func=None, n_trials=None):

        ## минимизируем ошибку
        self.study.optimize(lambda trial: self.objective(trial, X, y, cv, model, params_func),
                            n_trials=n_trials, callbacks=[optuna_utils.NeptuneCallback()])
    def lgbm_model(self,X, y, cv, params, trial):

        def lgb_scoring(y_hat, data):
            y_true = data.get_label()
            mape = np.mean(np.abs(y_true - y_hat)/np.abs(y_true))
            return 'loss', mape, False

        train_data = lgb.Dataset(X, y)
        cv_model = lgb.cv(params = params,
                          train_set = train_data,
                          folds = cv[:-1],
                          feval = lgb_scoring,
                          early_stopping_rounds = 10,
                          verbose_eval = False)

        X_train = X.iloc[cv[-1][0], :]
        y_train = y.iloc[cv[-1][0]]
        X_test = X.iloc[cv[-1][1], :]
        y_test = y.iloc[cv[-1][1]]
        train_data = lgb.Dataset(X_train,
                                 y_train,
                                 categorical_feature = [X.columns.get_loc(c) for c in \
                                                        [col for col in X.columns \
                                                         if ('popular' in col) | ('identifier' in col)]])
        test_data = lgb.Dataset(X_test,
                                y_test,
                                categorical_feature = [X.columns.get_loc(c) for c in \
                                                       [col for col in X.columns \
                                                        if ('popular' in col) | ('identifier' in col)]])
        evals_result = {}
        params['n_estimators'] = len(cv_model['loss-mean'])
        test_model = lgb.train(params = params,
                               train_set = train_data,
                               valid_sets=[test_data],
                               valid_names=['test_data'],
                               feval = lgb_scoring,
                               evals_result = evals_result,
                               verbose_eval = False)
#        feature_imp = pd.DataFrame({'Column': X.columns, 'Importance': test_model.feature_importance()})
#        feature_imp.to_csv('feature_imp_{}.csv'.format(trial.number), index = False)
#        neptune.log_artifact('feature_imp_{}.csv'.format(trial.number))
        test_loss = evals_result['test_data']['loss'][-1]
        return (cv_model, test_loss)

    def pl_model(self, X, y, cv, params, trial):

        batch_size = params['batch_size']
        params.pop('batch_size')
        iters = pd.DataFrame(columns=['B_C2H6', 'B_C3H8', 'B_iC4H10', 'B_nC4H10'])
        scores = pd.DataFrame(columns=['B_C2H6', 'B_C3H8', 'B_iC4H10', 'B_nC4H10'])
        fold_num = 0
        for fold in cv[:-1]:
            for target in ['B_C2H6', 'B_C3H8', 'B_iC4H10', 'B_nC4H10']:
                model_cv = self.training_nn(self.nn_model, X, y[[target]])
                model_cv.train(min_epochs=5,
                               max_epochs=3000,
                               model_params=params,
                               fold=fold,
                               batch_size=batch_size)
                trainer = model_cv.trainer
                iters.loc[fold_num, target] = float(trainer.current_epoch)
                scores.loc[fold_num, target] = float(trainer.callback_metrics['val_loss'].numpy())
            fold_num = fold_num+1
        best_iters = iters.mean(axis = 1).values
        best_cv = scores.mean(axis = 1).values

        test_loss = []
        for target in ['B_C2H6', 'B_C3H8', 'B_iC4H10', 'B_nC4H10']:

            mean_best_iter = round(iters[target].mean())
            model_test = self.training_nn(self.nn_model, X, y[[target]])
            test_losses = model_test.train(min_epochs=mean_best_iter,
                                           max_epochs=mean_best_iter,
                                           model_params=params,
                                           fold=cv[-1],
                                           val_fold=False,
                                           batch_size=batch_size)
            test_loss.append(float(model_cv.trainer.callback_metrics['val_loss'].numpy()))
        pp = 0
        for target in ['B_C2H6', 'B_C3H8', 'B_iC4H10', 'B_nC4H10']:
            neptune.log_text(target + '_cv_scores', str(list(scores[target].values)))
            neptune.log_metric(target + '_test_score', test_loss[pp])
            neptune.log_metric(target + '_run_score', np.mean(scores[target].values))
            neptune.log_metric(target + '_iters', np.mean(iters[target].values))
            pp = pp+1

        test_loss = np.mean(test_loss)

        return(np.mean(best_cv), np.std(best_cv), str(best_cv), np.mean(best_iters), test_loss)

    def objective(self, trial, X, y, cv, model, params_func):

        ## Множество параметров моделей
        params = params_func(trial, X)
        X_trans, y_trans, cv_trans, params_trans = self.preprocessing(X.copy(), y.copy(), copy.deepcopy(cv), params)

        if model == 'lgbm':
            cv_model, test_loss = self.lgbm_model(X_trans,
                                                  y_trans,
                                                  cv_trans,
                                                  params_trans,
                                                  trial)
            mean_cv = cv_model['loss-mean'][-1]
            iters = len(cv_model['loss-mean'])
            neptune.log_metric('std_cv_loss', cv_model['loss-stdv'][-1])
            neptune.log_metric('iterations', iters)
            neptune.log_metric('test_loss', test_loss)

        if model == 'torch':
            mean_cv, std_cv_loss, cv_scores, iterations, test_loss = self.pl_model(X_trans,
                                                                                   y_trans,
                                                                                   cv_trans,
                                                                                   params_trans,
                                                                                   trial)

            neptune.log_metric('std_cv_loss', std_cv_loss)
            neptune.log_text('cv_scores', cv_scores)
            neptune.log_metric('iterations', iterations)
            neptune.log_metric('test_loss', test_loss)

        return (mean_cv)

    def preprocessing(self, X, y, cv, params):

        print('current params:', params)
        n_back = params['n_back']
        params.pop('n_back')
        n_in = params['n_in']
        params.pop('n_in')

        print('X shape before dropped nans', X.shape)
        X = X.shift(n_back)
        X = X.dropna()
        X = X.reset_index(drop = True)
        print('X shape after dropped nans', X.shape)
        print('cv shape before dropped nans', len(cv[0][0]))

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
        y = y[n_in+n_back:]['B_C2H6']

        t = 0
        for fold in cv:
            fold[0] = fold[0][n_in+n_back:]
            for idx1 in range(0, len(fold[0])):
                fold[0][idx1] = fold[0][idx1] - n_in - n_back

            for idx2 in range(0, len(fold[1])):
                fold[1][idx2] = fold[1][idx2] - n_in - n_back
            cv[t] = fold
            t = t + 1
        print('cv shape after dropped nans', len(cv[0][0]))
        return X, y, cv, params