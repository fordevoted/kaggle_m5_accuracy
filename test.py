import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import utility

dataPath = "./kaggle/input"
dt = pd.read_csv(dataPath + "/sales_train_validation.csv")

def test1():
    dataPath = "./kaggle/input"

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
    del calendar

    event_type = pd.get_dummies(event_type)
    event2_type = pd.get_dummies(event2_type)
    for column in event2_type.columns:

        if column in event_type.columns:
            # print("pause")
            event_type[column] += event2_type[column]
        else:
            event2_type.index = event_type.index
            event_type = pd.concat([event_type, event2_type], axis=1)
    del event_type['0_0.0']
    dt = pd.read_csv(dataPath + "/sales_train_validation.csv")
    dt = dt.T
    dt = dt[6 + 350:]
    print(dt.head(3))
    print("==" * 30)
    # event_type.index = dt.index
    # dt = pd.concat([dt, event_type], axis=1)
    # "daysBeforeEventTest" will be used as input for predicting (We will forecast the days 1913-1941)
    daysBeforeEventTest = daysBeforeEvent[1913:1941]
    # "daysBeforeEvent" will be used for training as a feature.
    daysBeforeEvent = daysBeforeEvent[350:1913]

    event_typeTest = event_type.iloc[1913:1941]
    event_type = event_type[350:1913]

    event_type.index = dt.index
    dt = pd.concat([dt, event_type], axis=1)
    daysBeforeEvent.index = dt.index
    dt = pd.concat([dt, daysBeforeEvent], axis=1)

    timesteps = 14
    sc = MinMaxScaler(feature_range=(0, 1))
    dt_scaled = sc.fit_transform(dt)
    inputs = dt[-timesteps:]
    inputs = sc.transform(inputs)
    X_test = []
    X_test.append(inputs[0:timesteps])
    X_test = np.array(X_test)
    predictions = []
    y_train = []
    for i in range(timesteps, 1913 - 350):
        y_train.append(dt_scaled[i][0:30490])

    for j in range(timesteps, timesteps + 28):
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = (y_train[j]).reshape((1, 30490))
        testInput = np.column_stack((np.array(predicted_stock_price), daysBeforeEventTest[0][1913 + j - timesteps]))
        print(event_typeTest.iloc[j - timesteps])
        e = event_typeTest.iloc[j - timesteps].values
        testInput = np.column_stack((testInput, e.reshape((1, e.size))))
        X_test = np.append(X_test, testInput).reshape(1, j + 1, len(dt.columns))
        predicted_stock_price = sc.inverse_transform(testInput)[:, 0:30490]
        predictions.append(predicted_stock_price)

    print("end")


def test2(dt, startDay=350, finishDay=1913, savePath="./kaggle/save data"):

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
        data = utility.downcast_dtypes(data)
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
        if index % 100 == 99:
            print("## patch prices:" + str(index))
            break
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
        print("## data inter:" + str(i))
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
        data.insert(len(data.columns) - 1, 'prices', prices)
        for i in range(30490):
            value = data.at[i, 'prices']
            data.at[i, 'prices'] = value if not math.isnan(value) else mean_prices[i][0]
        index += 1
        data.to_csv(savePath + "/d_" + str(index) + "_data.csv", index=False)


test2(dt, 350, 1913, "./kaggle/training data")
