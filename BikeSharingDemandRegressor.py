import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV

df_train = pd.read_csv('dataset/train.csv')

df_test = pd.read_csv('dataset/test.csv')

df_test.rename(columns={'weathersit': 'weather',
                        'mnth': 'month',
                        'hr': 'hour',
                        'yr': 'year',
                        'hum': 'humidity',
                        'cnt': 'count'}, inplace=True)

df_train.rename(columns={'weathersit': 'weather',
                         'mnth': 'month',
                         'hr': 'hour',
                         'yr': 'year',
                         'hum': 'humidity',
                         'cnt': 'count'}, inplace=True)

df_train = df_train.drop(['atemp', 'windspeed', 'casual', 'registered'], axis=1)

df_test = df_test.drop(['atemp', 'windspeed'], axis=1)

df_train['month_year'] = df_train['month'] + df_train['year']
df_train['temp'] = np.sqrt(df_train['temp'])
df_train['humidity'] = np.square(df_train['humidity'])

y = df_train['count']
X = df_train[['season', 'year', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity',
              'month_year']]

train_or_submit = input("1. Train and Test the Model. \n2. Train and Submit. \n3. Perform RandomizedSearchCV. \n> ")

if train_or_submit == '1':

    print('Splitting Data -> 70% for Training and 30% for Testing.\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = np.log1p(y_train)

    # CatBoost
    print('Training with CatBoost... ')
    booster = CatBoostRegressor(learning_rate=0.02, iterations=2700, depth=9, silent=True)

    booster.fit(X_train, y_train)
    booster_pred = booster.predict(X_test)
    booster_pred[booster_pred < 0] = 0
    booster_pred = np.expm1(booster_pred)

    print('Done! ')
    print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, booster_pred.astype(int))))
    print('R2:', r2_score(y_test, booster_pred.astype(int)))

    # Random Forest
    print('\nTraining with Random Forest... ')

    rf = RandomForestRegressor(random_state=42, n_estimators=1600, min_samples_split=2, min_samples_leaf=1,
                               max_depth=50)

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_pred[rf_pred < 0] = 0
    rf_pred = np.expm1(rf_pred)

    print('Done!')
    print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, rf_pred.astype(int))))
    print('R2:', r2_score(y_test, rf_pred.astype(int)))

    # Ensemble
    y_pred = booster_pred * 0.9 + rf_pred * 0.1

    print('\nEnsemble:')
    print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, y_pred.astype(int))))
    print('R2:', r2_score(y_test, y_pred.astype(int)))

elif train_or_submit == '2':

    df_test['month_year'] = df_test['month'] + df_test['year']
    df_test['temp'] = np.sqrt(df_test['temp'])
    df_test['humidity'] = np.square(df_test['humidity'])
    y = np.log1p(y)

    # CatBoost
    print('Waiting for CatBoost... ')
    booster = CatBoostRegressor(learning_rate=0.02, iterations=2700, depth=9, silent=True)

    booster.fit(X, y)
    booster_pred = booster.predict(df_test)
    booster_pred[booster_pred < 0] = 0
    booster_pred = np.expm1(booster_pred)
    print('Done!\n')

    # Random Forest
    print('Waiting for Random Forest... ')
    rf = RandomForestRegressor(random_state=42, n_estimators=1600, min_samples_split=2, min_samples_leaf=1,
                               max_depth=50)

    rf.fit(X, y)
    rf_pred = rf.predict(df_test)
    rf_pred[rf_pred < 0] = 0
    rf_pred = np.expm1(rf_pred)

    # Ensemble
    y_pred = booster_pred * 0.9 + rf_pred * 0.1

    print('Done!\n')
    submission = pd.DataFrame()
    submission['Id'] = range(y_pred.shape[0])
    submission['Predicted'] = y_pred.astype(int)
    submission.to_csv("submission.csv", index=False)
    print('Made the submission file..')

elif train_or_submit == '3':

    boost_or_rf = input('1. CatBoost.\n2. Random Forest.\n> ')
    if boost_or_rf == '1':
        y = np.log1p(y)

        iterations = [int(x) for x in np.linspace(start=100, stop=4000, num=10)]
        learning_rate = [0.02, 0.05, 0.1, 0.3]
        depth = [4, 5, 6, 7, 8, 9, 10, 11, 12]
        random_grid = {'iterations': iterations,
                       'learning_rate': learning_rate,
                       'depth': depth
                       }

        boost = CatBoostRegressor(silent=True)

        boost_random = RandomizedSearchCV(estimator=boost, param_distributions=random_grid, n_iter=100, cv=3,
                                          verbose=2, random_state=42, n_jobs=-1)

        boost_random.fit(X, y)

        print(boost_random.best_params_)

        # BEST PARAMS = {'learning_rate': 0.02, 'iterations': 2700, 'depth': 9}

        chose_correct = True
    elif boost_or_rf == '2':
        y = np.log1p(y)

        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf = RandomForestRegressor()

        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3,
                                       verbose=2,
                                       random_state=42, n_jobs=-1)

        rf_random.fit(X, y)

        print(rf_random.best_params_)

        # BEST PARAMS = n_estimators=1400, min_samples_split=2, min_samples_leaf=1, max_features='auto',
        #                                    max_depth=100, bootstrap=True

        # Best Params Count = n_estimators =1600, min_samples_split = 2, min_samples_leaf = 1, max_depth = 50,
        #                                    bootstrap = True
