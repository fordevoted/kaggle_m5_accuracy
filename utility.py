import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv2D, Flatten
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


def preprocessing(dt, startDay=350, finishDay=1913, savePath="./kaggle/training data"):
    base_dt = dt[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    training_dt = []
    calendar = pd.read_csv(dataPath + "/calendar.csv")
    # Create dataframe with zeros for 1969 days in the calendar
    daysBeforeEvent = pd.DataFrame(np.zeros((1969, 1)))
    event_type = pd.DataFrame(np.zeros((1969, 1)))
    event2_type = pd.DataFrame(np.zeros((1969, 1)))

    # Extract feature from calendar including event 1 and 2
    print("## start extract event feature")
    for x, y in calendar.iterrows():
        if not (pd.isnull(calendar["event_name_1"][x])):
            daysBeforeEvent[0][x] += 1
            event_type[0][x] = calendar["event_type_1"][x]
        if not (pd.isnull(calendar["event_name_2"][x])):
            daysBeforeEvent[0][x] += 1
            event2_type[0][x] = calendar["event_type_2"][x]
            # if first day was an event this row will cause an exception because "x-1".
            # Since it is not i did not consider for now.
    week_day = calendar['weekday']
    # not use any more
    print("## start to combine event1 and 2")
    event_type = pd.get_dummies(event_type)
    event2_type = pd.get_dummies(event2_type)
    week_day = pd.get_dummies(week_day)

    for column in event2_type.columns:
        if column in event_type.columns:
            event_type[column] += event2_type[column]
        else:
            event2_type.index = event_type.index
            event_type = pd.concat([event_type, event2_type], axis=1)
    del event_type['0_0.0']
    del event2_type

    event_type = event_type[startDay:finishDay]
    week_day = week_day[startDay:finishDay]
    daysBeforeEvent = daysBeforeEvent[startDay:finishDay]

    # create data in time series
    for i in range(startDay, finishDay):
        training_dt.append(base_dt.copy())

    # Before concatenation with our main data "dt", indexes are made same and column name is changed to
    # "oneDayBeforeEvent"
    # calendar
    print("## start insert feature for training data")
    index = 0
    for data in training_dt:
        data.insert(len(data.columns), 'eventForThisDay', daysBeforeEvent.loc[startDay + index, 0])
        for col in event_type.columns:
            data.insert(len(data.columns), col, event_type.at[startDay + index, col])
        for col in week_day.columns:
            data.insert(len(data.columns), col, week_day.at[startDay + index, col])
        data.insert(len(data.columns), 'snap_CA', calendar.at[startDay + index, 'snap_CA'])
        data.insert(len(data.columns), 'snap_TX', calendar.at[startDay + index, 'snap_TX'])
        data.insert(len(data.columns), 'snap_WI', calendar.at[startDay + index, 'snap_WI'])
        index += 1
        data = downcast_dtypes(data)
    del calendar, event_type, week_day, daysBeforeEvent
    # print(training_dt[1553].head(10))

    prices = pd.read_csv(dataPath + "/sell_prices.csv")
    # initial create columns
    goodsIndex = 0
    store_id_past = ""
    item_id_past = ""
    for data in training_dt:
        data['prices'] = np.nan

    for index, row in prices.iterrows():
        if index % 10000 == 0:
            print("## patch prices:" + str(index))
            # break
        # 11101 is initial week which also known as day 1
        # row is store_id   item_id     wm_yr_wk	sell_price
        day_index = (int(row[2]) - 11101) * 7 + 1
        isInitialIndex = False

        for day in range(day_index, day_index + 7):
            if day < startDay or day > finishDay:
                continue
            else:
                data = training_dt[day - startDay]

                if (not isInitialIndex) and row[0] == store_id_past and row[1] == item_id_past:
                    isInitialIndex = True
                elif not isInitialIndex:
                    goodsIndex = data[(data['store_id'] == row[0]) & (data['item_id'] == row[1])].index
                    store_id_past = row[0]
                    item_id_past = row[1]
                    isInitialIndex = True
                data.at[goodsIndex, 'prices'] = row[3]
    print("## start to calculate average prices to fill NAN")
    mean_prices = np.zeros((30490, 1))
    count_prices = np.zeros((30490, 1))
    for i in range(30490):
        # print("## data inter:" + str(i))
        for data in training_dt:
            value = data.at[i, 'prices']
            if not math.isnan(value):
                mean_prices[i][0] += value
                count_prices[i][0] += 1
        mean_prices[i][0] = (mean_prices[i][0] / count_prices[i][0]) if count_prices[i][0] > 0 else 0

    index = 0
    print("##　start to save　data")
    for data in training_dt:
        del data['item_id']
        data = pd.get_dummies(data)
        prices = data['prices']
        data.drop(labels=['prices'], axis=1, inplace=True)
        data.insert(len(data.columns), 'prices', prices)
        for i in range(30490):
            value = data.at[i, 'prices']
            data.at[i, 'prices'] = value if not math.isnan(value) else mean_prices[i][0]
        index += 1
        data.to_csv(savePath + "/d_" + str(index) + "_data.csv", index=False)


def prepare_training_data(startDay=350, data_path="./kaggle/training data"):
    # Feature Scaling
    # Scale the features using min-max scaler in range 0-1
    training_data = []
    # TODO correct data count
    # count = 15;
    for dirPath, dirNames, fileNames in os.walk(data_path):
        for f in fileNames:
            # if count < 0:
            #     break

            training_data.append(pd.read_csv(os.path.join(dirPath, f)))
            # count -= 1
    dt = pd.read_csv(dataPath + "/sales_train_validation.csv")
    print("## loading data complete")
    index = 0
    for data in training_data:
        dt.index = data.index
        data['sales'] = dt['d_' + str(startDay + index)]
        prices_mean = data['prices'].mean(skipna=True)
        data['prices'] = data.prices.mask(data.prices ==0, prices_mean)
        training_data.pop(index)
        training_data.insert(index, np.array(data))
        index += 1
        data = downcast_dtypes(data)
    print("## complete patch sales")
    X_train = []
    y_train = []
    ## TODO correct count
    for i in range(timesteps, 1913 - timesteps):
        data_list = training_data[i - timesteps:i]
        x = data_list[0]
        x = np.dstack((x, data_list[1]))
        for data in data_list[2:]:
            x = np.concatenate((x, data[:, :, None]), axis=-1)
        x = x.reshape((data_list[0].shape[0], data_list[0].shape[1], timesteps))
        # print(x.shape)
        X_train.append(x)
        y_train.append(dt[:]['d_' + str(startDay + i + 1)])
        # İmportant!! if extra features are added (like oneDayBeforeEvent)
        # use only sales values for predictions (we only predict sales)
        # this is why 0:30490 columns are choosen
    print("## complete prepare training data")
    del dt, training_data
    # Convert to np array to be able to feed the LSTM model
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print("x_train_shape", X_train.shape)
    print("y_train_shape", y_train.shape)
    return X_train, y_train


def create_model(x_train_shape):
    # model
    # print(x_train_shape)
    # Initialising the RNN
    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    layer_1_units = 50
    model.add(Conv2D(14, kernel_size=3, strides=(20, 2), input_shape=(x_train_shape[1], x_train_shape[2], x_train_shape[3])))
    model.add(Conv2D(8, kernel_size=3, strides=(10, 2)))
    model.add(Conv2D(6, kernel_size=3, strides=(5, 2)))
    model.add(Conv2D(3, kernel_size=3, strides=(2, 2)))
    # model.add(LSTM(units=layer_1_units, return_sequences=True))
    # model.add(Dropout(0.2))
    # layer_1_units = 256
    # model.add(LSTM(units=layer_1_units, return_sequences=True))
    # model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    layer_2_units = 400
    # model.add(LSTM(units=layer_2_units, return_sequences=True))
    # model.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    layer_3_units = 400
    # model.add(LSTM(units=layer_3_units))
    # model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30490, activation='relu'))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mean_squared_error"])
    model.summary()
    return model


# TODO predict
def predict(model, startDay, data_path, training_dataset, bias=350):
    # predict
    x_test = []

    for dirPath, dirNames, fileNames in os.walk(training_dataset):
        for f in fileNames:
            end = f.find('_data.csv',)
            day = int(f[2:end])
            if day > (startDay - timesteps - bias):
                x_test.append(pd.read_csv(os.path.join(dirPath, f)))
    print("## end of reading  data for 14 day in training data, x_test size:", len(x_test))
    dt = pd.read_csv(dataPath + "/sales_train_validation.csv")
    index = 0
    for data in x_test:
        dt.index = data.index
        data['sales'] = dt['d_' + str(startDay - timesteps + index)]
        # inplace store
        x_test.pop(index)
        x_test.insert(index, np.array(data))
        index += 1
    del dt
    print("## end of patching sales")
    for dirPath, dirNames, fileNames in os.walk(data_path):
        for f in fileNames:
            x_test.append(np.array(pd.read_csv(os.path.join(dirPath, f))))
    print("## end of reading testing data, x_test size : ", len(x_test))
    predictions = []
    for j in range(timesteps, timesteps + 28):
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        data_list = x_test[j - timesteps:j]
        # print(j - timesteps, j)
        x = data_list[0]
        x = np.dstack((x, data_list[1]))

        for data in data_list[2:]:
            # print(j, x.shape, data.shape)
            x = np.concatenate((x, data[:, :, None]), axis=-1)
        x = x.reshape((1, data_list[0].shape[0], data_list[0].shape[1], timesteps))
        predicted_stock_price = np.squeeze(model.predict(x), axis=0)
        x_test[j] = np.concatenate((x_test[j], np.array(predicted_stock_price)[:, None]), axis=1).reshape(30490, x_test[0].shape[1])
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
