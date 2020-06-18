import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os
from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dataPath = "./kaggle/input"
timesteps = 14
startDay = 350
sc = MinMaxScaler(feature_range=(0, 1))



def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def preprocessing(dt):
    dt = dt.T
    dt.head(8)
    # Remove id, item_id, dept_id, cat_id, store_id, state_id columns
    base_dt = dt[2:6]
    days_dt = dt[6 + startDay:]
    dt = dt[6 + startDay:]
    dt.head(5)
    calendar = pd.read_csv(dataPath + "/calendar.csv")
    # Create dataframe with zeros for 1969 days in the calendar
    daysBeforeEvent = pd.DataFrame(np.zeros((1969, 1)))
    event_type = pd.DataFrame(np.zeros((1969, 1)))
    event2_type = pd.DataFrame(np.zeros((1969, 1)))

    # "1" is assigned to the days before the event_name_1. Since "event_name_2" is rare, it was not added.
    for x, y in calendar.iterrows():
        if not (pd.isnull(calendar["event_name_1"][x])):
            daysBeforeEvent[0][x - 1] += 1
            event_type[0][x - 1] = calendar["event_type_1"][x]
        if not (pd.isnull(calendar["event_name_2"][x])):
            daysBeforeEvent[0][x - 1] += 1
            event2_type[0][x - 1] = calendar["event_type_2"][x]
            # if first day was an event this row will cause an exception because "x-1".
            # Since it is not i did not consider for now.
    # "calendar" won't be used anymore.
    week_day = calendar['weekday']
    # del calendar

    event_type = pd.get_dummies(event_type)
    event2_type = pd.get_dummies(event2_type)
    week_day = pd.get_dummies(week_day)

    for column in event2_type.columns:
        if column in event_type.columns:
            # print("pause")
            event_type[column] += event2_type[column]
        else:
            event2_type.index = event_type.index
            event_type = pd.concat([event_type, event2_type], axis=1)
    del event_type['0_0.0']
    # "daysBeforeEventTest" will be used as input for predicting (We will forecast the days 1913-1941)
    daysBeforeEventTest = daysBeforeEvent[1913:1941]
    event_typeTest = event_type.iloc[1913:1941]
    week_dayTest = week_day.iloc[1913:1941]

    event_type = event_type[startDay:1913]
    week_day = week_day[startDay:1913]
    # "daysBeforeEvent" will be used for training as a feature.
    daysBeforeEvent = daysBeforeEvent[startDay:1913]
    # Before concatanation with our main data "dt", indexes are made same and column name is changed to "oneDayBeforeEvent"
    daysBeforeEvent.columns = ["oneDayBeforeEvent"]

    daysBeforeEvent.index = dt.index
    dt = pd.concat([dt, daysBeforeEvent], axis=1)
    event_type.index = dt.index
    dt = pd.concat([dt, event_type], axis=1)
    week_day.index = dt.index
    dt = pd.concat([dt, week_day], axis=1)
    return dt, daysBeforeEventTest, event_typeTest, week_dayTest


def prepare_training_data(dt):
    # Feature Scaling
    # Scale the features using min-max scaler in range 0-1

    dt_scaled = sc.fit_transform(dt)
    X_train = []
    y_train = []
    for i in range(timesteps, 1913 - startDay):
        X_train.append(dt_scaled[i - timesteps:i])
        y_train.append(dt_scaled[i][0:30490])
        # İmportant!! if extra features are added (like oneDayBeforeEvent)
        # use only sales values for predictions (we only predict sales)
        # this is why 0:30490 columns are choosen
    del dt_scaled
    # Convert to np array to be able to feed the LSTM model
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train


def create_model(x_train_shape):
    # model
    # Initialising the RNN
    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    layer_1_units = 50
    model.add(LSTM(units=layer_1_units, return_sequences=True, input_shape=(x_train_shape[1], x_train_shape[2])))
    model.add(Dropout(0.2))

    # layer_1_units = 256
    # model.add(LSTM(units=layer_1_units, return_sequences=True))
    # model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    layer_2_units = 400
    model.add(LSTM(units=layer_2_units, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    layer_3_units = 400
    model.add(LSTM(units=layer_3_units))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=30490))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mean_squared_error"])
    return model


def predict(model, dt, daysBeforeEventTest, event_typeTest, week_dayTest):
    # predict
    inputs = dt[-timesteps:]
    inputs = sc.transform(inputs)
    X_test = []
    X_test.append(inputs[0:timesteps])
    X_test = np.array(X_test)
    predictions = []

    for j in range(timesteps, timesteps + 28):
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = model.predict(X_test[0, j - timesteps:j].reshape(1, timesteps, len(dt.columns)))
        testInput = np.column_stack((np.array(predicted_stock_price), daysBeforeEventTest[0][1913 + j - timesteps]))
        e = event_typeTest.iloc[j - timesteps].values
        testInput = np.column_stack((testInput, e.reshape((1, e.size))))
        w = week_dayTest.iloc[j - timesteps].values
        testInput = np.column_stack((testInput, w.reshape((1, w.size))))
        X_test = np.append(X_test, testInput).reshape(1, j + 1, len(dt.columns))
        predicted_stock_price = sc.inverse_transform(testInput)[:, 0:30490]
        predictions.append(predicted_stock_price)
    return predictions


def to_submission(predictions):
    # to submission
    submission = pd.DataFrame(data=np.array(predictions).reshape(28, 30490))
    submission = submission.T
    submission = pd.concat((submission, submission), ignore_index=True)
    sample_submission = pd.read_csv(dataPath + "/sample_submission.csv")
    idColumn = sample_submission[["id"]]
    submission[["id"]] = idColumn
    cols = list(submission.columns)
    cols = cols[-1:] + cols[:-1]
    submission = submission[cols]
    colsdeneme = ["id"] + [f"F{i}" for i in range(1, 29)]
    submission.columns = colsdeneme
    submission.to_csv("submission.csv", index=False)
