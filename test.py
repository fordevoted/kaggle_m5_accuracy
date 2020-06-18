import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



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
        event_type[0][x-1] = calendar["event_type_1"][x]
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
print("=="*30)
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
# Before concatanation with our main data "dt", indexes are made same and column name is changed to "oneDayBeforeEvent"
#
# print(len(dt.columns))
# print(dt.head(3))
