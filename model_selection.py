import pandas as pd
import lightgbm as lgb
import neptune
import optuna
import neptunecontrib.monitoring.optuna as optuna_utils
import numpy as np
from sklearn.metrics import mean_absolute_error
import gc
import torch
import copy

class training:

    def __init__(self, nn_model = None, training_nn = None,
                 name=None, description=None, params = None, properties=None, tags=None, upload_source_files=None):

        ## Создаем эксперимент в neptune
        neptune.create_experiment(name, description, params, properties, tags, upload_source_files)
        self.scoring_func = mean_absolute_error
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
            return 'loss', self.scoring_func(y_true, y_hat), False

        train_data = lgb.Dataset(X, y)
        cv_model = lgb.cv(params = params,
                          train_set = train_data,
                          folds = cv[:-1],
                          feval = lgb_scoring,
                          early_stopping_rounds = 20,
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
        feature_imp = pd.DataFrame({'Column': X.columns, 'Importance': test_model.feature_importance()})
        feature_imp.to_csv('feature_imp_{}.csv'.format(trial.number), index = False)
        neptune.log_artifact('feature_imp_{}.csv'.format(trial.number))
        test_loss = evals_result['test_data']['loss'][-1]
        return (cv_model, test_loss)

    def pl_model(self, X, y, cv, params, trial):

        print('current params:', params)
        seq_len = params['seq_len']
        params.pop('seq_len')
        batch_size = params['batch_size']
        params.pop('batch_size')

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
        y = y[(seq_len-1):]

        best_iters = []
        best_cv = []

        t = 0
        for fold in cv:
            fold[0] = fold[0][(seq_len-1):]
            for idx1 in range(0, len(fold[0])):
                fold[0][idx1] = fold[0][idx1] - seq_len + 1

            for idx2 in range(0, len(fold[1])):
                fold[1][idx2] = fold[1][idx2] - seq_len + 1
            cv[t] = fold
            t = t + 1

        print(cv[0][0])
        print(X.shape)
        print(y.shape)
        for fold in cv[:-1]:
            model_cv = self.training_nn(self.nn_model, X, y)
            model_cv.train(min_epochs=5,
                           max_epochs=3000,
                           model_params=params,
                           fold=fold,
                           batch_size=batch_size)
            val_losses = model_cv.model.val_losses
            best_iters.append(len(val_losses))
            best_cv.append(val_losses[-1])


        mean_best_iter = round(np.mean(best_iters))
        model_test = self.training_nn(self.nn_model, X, y)
        test_losses = model_test.train(min_epochs=mean_best_iter,
                                       max_epochs=mean_best_iter,
                                       model_params=params,
                                       fold=cv[-1],
                                       val_fold=False,
                                       batch_size=batch_size)
        test_losses = model_test.model.val_losses
        test_loss = test_losses[-1]
        return(np.mean(best_cv), np.std(best_cv), str(best_cv), np.mean(best_iters), test_loss)

    def objective(self, trial, X, y, cv, model, params_func):

        ## Множество параметров моделей
        params = params_func(trial, X)

        if model == 'lgbm':
            cv_model, test_loss = self.lgbm_model(X.copy(),
                                                  y.copy(),
                                                  copy.deepcopy(cv),
                                                  params,
                                                  trial)
            mean_cv = len(cv_model['loss-mean'])
            neptune.log_metric('std_cv_loss', cv_model['loss-stdv'][-1])
            neptune.log_metric('iterations', mean_cv )
            neptune.log_metric('test_loss', test_loss)

        if model == 'torch':
            mean_cv, std_cv_loss, cv_scores, iterations, test_loss = self.pl_model(X.copy(),
                                                                                   y.copy(),
                                                                                   copy.deepcopy(cv),
                                                                                   params,
                                                                                   trial)

            neptune.log_metric('std_cv_loss', std_cv_loss)
            neptune.log_text('cv_scores', cv_scores)
            neptune.log_metric('iterations', iterations)
            neptune.log_metric('test_loss', test_loss)
        return(mean_cv)