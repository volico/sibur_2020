import pandas as pd
import pickle
from flask import Flask, request


model_iC4H10 = pickle.load(open('models/iC4H10.pickle', 'rb'))
model_C2H6 = pickle.load(open('models/C2H6.pickle', 'rb'))
model_C3H8 = pickle.load(open('models/C3H8.pickle', 'rb'))
model_nC4H10 = pickle.load(open('models/nC4H10.pickle', 'rb'))
data = {'X_iC4H10': pd.read_csv('data/X_iC4H10.csv', parse_dates = ['timestamp']),
        'X_C2H6': pd.read_csv('data/X_C2H6.csv', parse_dates = ['timestamp']),
        'X_C3H8': pd.read_csv('data/X_C3H8.csv', parse_dates = ['timestamp']),
        'X_nC4H10': pd.read_csv('data/X_nC4H10.csv', parse_dates = ['timestamp'])}

app = Flask(__name__)

@app.route('/predict')
def make_prediction():
    '''Делает предсказание состава ШФЛУ на заданный в запросе промежуток времени
    Возвращает предсказания в виде json
    '''

    posted_data = request.get_json()
    try:
        start_time = pd.to_datetime(posted_data['start_time'])
        end_time = pd.to_datetime(posted_data['end_time'])
        wrong_format = False
    except:
        wrong_format = True

    if wrong_format == True:
        return 'Неправильный формат данных. Необходим json формата {"start_time": {год-месяц-число часы:минуты:секунды}, \
        "end_time": {год-месяц-число часы:минуты:секунды}}'

    elif (start_time < pd.to_datetime('2020-02-19 02:30:00')) | (end_time < pd.to_datetime('2020-02-19 02:30:00')):
        return 'start_time и end_time должны быть >= 2020-02-19 02:30:00'

    elif (start_time > pd.to_datetime('2020-07-22 23:30:00')) | (end_time > pd.to_datetime('2020-07-22 23:30:00')):
        return 'start_time и end_time должны быть <= 2020-07-22 23:30:00'

    elif start_time > end_time:
        return 'start_time должна быть меньше end_time'

    else:
        '''Если все условия выполнены, то выполняется предсказание
        '''

        predictions = pd.DataFrame()
        for target, model in zip(['iC4H10', 'C2H6', 'C3H8', 'nC4H10'],
                                 [model_iC4H10, model_C2H6, model_C3H8, model_nC4H10]):

            X = data['X_{}'.format(target)]
            predictions[target] = model.predict(X[(X['timestamp'] >= start_time) &
                                                    (X['timestamp'] <= end_time)].drop('timestamp', axis = 'columns'))

        return predictions.to_json()

@app.route('/')
def about():
    return 'Для предсказания состава ШФЛУ на определенную дату необходимо сделать запрос на /predict и передать \
    необходимый промежуток предсказаний в виде json формата {"start_time": {год-месяц-число часы:минуты:секунды}, \
    "end_time": {год-месяц-число часы:минуты:секунды}}. Доступный промежуток 2020-02-19 02:30:00 - 2020-07-22 23:30:00. \
    Возвращает предсказания в виде json объекта'

app.run(host ='0.0.0.0')