import pandas as pd
import neptune
import optuna
import neptunecontrib.monitoring.optuna as optuna_utils
import numpy as np
import copy
import lightgbm as lgb
import preprocessing
import os
import catboost
import xgboost
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


class training:
    '''Класс, отвечающий за тренировку модели
    nn_model - модель нейронной сети, наследующая pl.LightningModule (при необходимости)
    trainning_nn - класс, тренирующий nn_model (при необходимости)
    остальные параметры - настройки эксперимента в neptune.ai (при необходимости)
    '''
    def __init__(
            self, nn_model = None, training_nn = None, sklearn_class = None,
            name=None, description=None, params = None,
            properties=None, tags=None, upload_source_files=None
    ):

        ## Создаем эксперимент в neptune
        neptune.create_experiment(name, description, params, properties, tags, upload_source_files)
        self.nn_model = nn_model
        self.training_nn = training_nn
        self.sklearn_class = sklearn_class

    def set_up_studying(self, random_state, direction='minimize'):
        '''Инициализация подборки параметров в optuna
        random_state - random seed для сэмплера параметров
        direction - направление оптимизации (максимизация или минимизация)
        '''

        ## Выбираем стандартный сэмплер для подборки параметров
        sampler = optuna.samplers.TPESampler(seed=random_state)

        self.study = optuna.create_study(sampler=sampler, direction=direction)

    def train(self, X, y = None, cv=None, model=None, params_func=None, n_trials=None):
        '''Подбор параметров в модели
        X - фичи (может быть датафрейм/матрица, в зависимости от preprocessing)
        y - таргеты (может быть датафрейм/матрица, в зависимости от preprocessing)
        cv - список фолдов для кросс валидации
        model - модель ('catboost', 'lgbm' или 'pytorch')
        params_func - фунция, возращающая для данного trial множество параметров
        n_trials - количество итераций при подборке параметров
        '''
        ## минимизируем ошибку
        self.study.optimize(lambda trial: self.objective(trial, X, y, cv, model, params_func),
                            n_trials=n_trials, callbacks=[optuna_utils.NeptuneCallback()])
    def lgbm_model(self,X, y, cv, params, log_importance, trial):
        '''Тренировка и оценка lgbm модели
        X - фичи (матрица/датафрейм, не важно)
        y - таргеты (матрица/датафрейм, не важно)
        cv - список фолдов для кросс валидации
        params - параметры для lgbm модели (и только для нее)
        log_importance - нужно ли логировать feature_importance в neptune
        trial - текущая итерация подбори параметров
        '''
        def lgb_scoring(y_hat, data):
            '''
            Функция для оценивания качества модели на валидационной выборке
            Возвращает: название метрики, метрика, is_high_better
            '''
            y_true = data.get_label()
            return 'loss', np.mean(np.abs((y_true - y_hat)/y_true)), False

        train_data = lgb.Dataset(X, y)
        cv_model = lgb.cv(params = params,
                          train_set = train_data,
                          folds = cv[:-1],
                          feval = lgb_scoring,
                          early_stopping_rounds = 10,
                          verbose_eval = False)

        X_train = X[cv[-1][0], :]
        y_train = y[cv[-1][0]]
        X_test = X[cv[-1][1], :]
        y_test = y[cv[-1][1]]
        train_data = lgb.Dataset(X_train,
                                 y_train)
        test_data = lgb.Dataset(X_test,
                                y_test
                                )
        evals_result = {}
        params['n_estimators'] = round(len(cv_model['loss-mean']))
        test_model = lgb.train(params = params,
                               train_set = train_data,
                               valid_sets=[test_data],
                               valid_names=['test_data'],
                               feval = lgb_scoring,
                               evals_result = evals_result,
                               verbose_eval = False)
        if log_importance == True:
            feature_imp = pd.DataFrame({'Column': ['A_rate', 'A_CH4', 'A_C2H6', 'A_C3H8', 'A_iC4H10',
       'A_nC4H10', 'A_iC5H12', 'A_nC5H12', 'A_C6H14', 'B_rate'], 'Importance': test_model.feature_importance()})
            feature_imp.to_csv('feature_imp_{}.csv'.format(trial.number), index = False)
            neptune.log_artifact('feature_imp_{}.csv'.format(trial.number))
            os.remove('feature_imp_{}.csv'.format(trial.number))
        test_loss = evals_result['test_data']['loss'][-1]
        return (cv_model, test_loss)


    def xgboost_model(self,X, y, cv, params, log_importance, trial):
        global cv_model
        '''Тренировка и оценка lgbm модели
        X - фичи (матрица/датафрейм, не важно)
        y - таргеты (матрица/датафрейм, не важно)
        cv - список фолдов для кросс валидации
        params - параметры для lgbm модели (и только для нее)
        log_importance - нужно ли логировать feature_importance в neptune
        trial - текущая итерация подбори параметров
        '''
        def xgb_scoring(y_hat, data):
            '''
            Функция для оценивания качества модели на валидационной выборке
            Возвращает: название метрики, метрика, is_high_better
            '''
            y_true = data.get_label()
            return 'loss', np.mean(np.abs((y_true - y_hat)/y_true))

        train_data = xgboost.DMatrix(X, y)
        num_boost_round = params['n_estimators']
        params.pop('n_estimators')
        cv_model = xgboost.cv(params = params,
                              dtrain = train_data,
                              folds = cv[:-1],
                              num_boost_round = num_boost_round,
                              maximize = False,
                              feval = xgb_scoring,
                              early_stopping_rounds = 10,
                              verbose_eval = False)
        X_train = X[cv[-1][0], :]
        y_train = y[cv[-1][0]]
        X_test = X[cv[-1][1], :]
        y_test = y[cv[-1][1]]
        train_data = xgboost.DMatrix(X_train,
                                     y_train)
        test_data = xgboost.DMatrix(X_test,
                                    y_test)
        evals_result = {}
        params['n_estimators'] = round(len(cv_model['test-loss-mean']))
        num_boost_round = params['n_estimators']
        params.pop('n_estimators')
        test_model = xgboost.train(params = params,
                                   dtrain = train_data,
                                   evals=[(test_data, 'test_data')],
                                   num_boost_round = num_boost_round,
                                   maximize=False,
                                   feval = xgb_scoring,
                                   evals_result = evals_result,
                                   verbose_eval = False)
        if log_importance == True:
            feature_imp = pd.DataFrame({'Column': ['A_rate', 'A_CH4', 'A_C2H6', 'A_C3H8', 'A_iC4H10',
       'A_nC4H10', 'A_iC5H12', 'A_nC5H12', 'A_C6H14', 'B_rate'], 'Importance': test_model.feature_importance()})
            feature_imp.to_csv('feature_imp_{}.csv'.format(trial.number), index = False)
            neptune.log_artifact('feature_imp_{}.csv'.format(trial.number))
            os.remove('feature_imp_{}.csv'.format(trial.number))
        test_loss = evals_result['test_data']['loss'][-1]
        return (cv_model, test_loss)


    def catboost_model(self, X, y, cv, params):
        # ЕЩЕ НЕДОПИСАНО
        '''Тренировка и оценка catboost модели ()
        X - фичи (матрица/датафрейм, не важно)
        y - таргеты (матрица/датафрейм, не важно)
        cv - список фолдов для кросс валидации
        params - параметры для lgbm модели (и только для нее)
        '''

        train_data = catboost.Pool(X, y)
        cv_model = catboost.cv(params = params,
                          pool = train_data,
                          folds = cv[:-1],
                          early_stopping_rounds = 10,
                          verbose_eval = False)

        X_train = X[cv[-1][0], :]
        y_train = y[cv[-1][0]]
        X_test = X[cv[-1][1], :]
        y_test = y[cv[-1][1]]
        train_data = catboost.Pool(X_train,
                                 y_train)
        test_data = catboost.Pool(X_test,
                                y_test
                                )

        metric_data = pd.read_csv('catboost_info/fold_0_test_error.tsv', sep='\t')
        last_mape = metric_data[['MAPE', 'MAPE.1']].values[-1]
        cv_mean = np.mean(last_mape)
        cv_std = np.std(last_mape)
        cv_iters = len(metric_data)
        params['n_estimators'] = cv_iters
        test_model = catboost.train(params = params,
                                    dtrain = train_data,
                                    verbose_eval = False)
        predictions = np.array(test_model.predict(test_data))
        y_test = y_test.values
        test_mape = np.mean(np.abs((y_test-predictions)/y_test))

        return (cv_mean, cv_std, cv_iters, test_mape)

    def sklearn_model(self, X, y, cv, params):

        def mape(y_true, y_pred):
            mape = np.mean(np.abs((y_true - y_pred) / y_true))
            return mape

        model = self.sklearn_class(**params)
        cv_output = cross_validate(model,
                                   X = X,
                                   y = y,
                                   cv = cv,
                                   scoring = make_scorer(mape, greater_is_better=False),
#                                   fit_params = {'sample_weight': 1/y[cv[0][0]]}
                                   )
        cv_score = cv_output['test_score']
        cv_mean = np.mean(cv_score[:-1])*(-1)
        cv_std = np.std(cv_score[:-1])
        test_mape = cv_score[-1]*(-1)

        return (cv_mean, cv_std, test_mape)


    def pl_model(self, X, y, cv, params):
        '''Тренировка и оценка нейронной сети на пайторче модели
        X - фичи (матрица/датафрейм, не важно)
        y - таргеты (матрица/датафрейм, не важно)
        cv - список фолдов для кросс валидации
        params - параметры для lgbm модели (и только для нее)
        '''


        batch_size = params['batch_size']
        params.pop('batch_size')
        iters = []
        scores = []
        fold_num = 0
        for fold in cv[:-1]:
            model_cv = self.training_nn(self.nn_model, X, y)
            model_cv.train(min_epochs=5,
                           max_epochs=3000,
                           model_params=params,
                           fold=fold,
                           batch_size=batch_size)
            trainer = model_cv.trainer
            iters.append(trainer.current_epoch)
            scores.append(trainer.callback_metrics['val_loss'].numpy())
            fold_num = fold_num+1
        best_iters = round(np.mean(iters))
        best_cv = scores

        model_test = self.training_nn(self.nn_model, X, y)
        model_test.train(min_epochs=best_iters,
                         max_epochs=best_iters,
                         model_params=params,
                         fold=cv[-1],
                         val_fold=False,
                         batch_size=batch_size)
        test_loss = float(model_test.trainer.callback_metrics['val_loss'].numpy())

        return(np.mean(best_cv), np.std(best_cv), best_iters, test_loss)

    def objective(self, trial, X, y, cv, model, params_func):
        '''Тренировка модели и логирование результатов
        X - фичи (матрица/датафрейм, не важно)
        y - таргеты (матрица/датафрейм, не важно)
        cv - список фолдов для кросс валидации
        model - модель ('catboost', 'lgbm' или 'pytorch')
        params_func - фунция, возращающая для данного trial множество параметров
        '''


        ## Множество параметров моделей
        params = params_func(trial, X)
        neptune.log_text('current_params', str(params))
        X_trans, y_trans, cv_trans, params_trans = preprocessing.preprocessing(X.copy(),
                                                                               y.copy(),
                                                                               copy.deepcopy(cv),
                                                                               copy.deepcopy(params))

        if model == 'lgbm':
            cv_model, test_loss = self.lgbm_model(X_trans,
                                                  y_trans,
                                                  cv_trans,
                                                  params_trans,
                                                  False,
                                                  trial)
            mean_cv = cv_model['loss-mean'][-1]
            iters = len(cv_model['loss-mean'])
            neptune.log_metric('std_cv_loss', cv_model['loss-stdv'][-1])
            neptune.log_metric('iterations', iters)
            neptune.log_metric('test_loss', test_loss)

        if model == 'xgboost':
            cv_model, test_loss = self.xgboost_model(X_trans,
                                                     y_trans,
                                                     cv_trans,
                                                     params_trans,
                                                     False,
                                                     trial)
            mean_cv = cv_model['test-loss-mean'].values[-1]
            iters = len(cv_model['test-loss-mean'])
            neptune.log_metric('std_cv_loss', cv_model['test-loss-std'].values[-1])
            neptune.log_metric('iterations', iters)
            neptune.log_metric('test_loss', test_loss)

        if model == 'catboost':
            mean_cv, cv_std, cv_iters, test_mape = self.catboost_model(X_trans,
                                                                       y_trans,
                                                                       cv_trans,
                                                                       params_trans)
            neptune.log_metric('std_cv_loss', cv_std)
            neptune.log_metric('iterations', cv_iters)
            neptune.log_metric('test_loss', test_mape)


        if model == 'torch':
            mean_cv, std_cv_loss, iterations, test_loss = self.pl_model(X_trans,
                                                                        y_trans,
                                                                        cv_trans,
                                                                        params_trans)

            neptune.log_metric('std_cv_loss', std_cv_loss)
            neptune.log_metric('iterations', iterations)
            neptune.log_metric('test_loss', test_loss)

        if model == 'sklearn':

            mean_cv, cv_std, test_loss = self.sklearn_model(X_trans, y_trans, cv_trans, params_trans)
            print(X_trans)

            neptune.log_metric('std_cv_loss', cv_std)
            neptune.log_metric('test_loss', test_loss)

        return (mean_cv)