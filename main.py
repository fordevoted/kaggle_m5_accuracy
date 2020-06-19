import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import utility

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dataPath = "./kaggle/input"
training_data_path = "./kaggle/training data"
test_data_path = "./kaggle/test data"
dt = pd.read_csv(dataPath + "/sales_train_validation.csv")
dt.head(3)
# Reduce memory usage and compare with the previous one to be sure
dt = utility.downcast_dtypes(dt)
# prepare
utility.preprocessing(dt, startDay=350, finishDay=1913, savePath=training_data_path)
# print(dt.columns)

X_train, y_train = utility.prepare_training_data(data_path=training_data_path)
model = utility.create_model(X_train.shape)

# hyper parameter
epoch = 32
batch_size = 44
validation_spilt = 0.2
filepath = "weights_best.hdf5"
early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=2)
checkPoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True,
                             mode='min')
# fit
model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,
          validation_split=validation_spilt, verbose=1, callbacks=[early_stopping, checkPoint])

# predict
#prepare test data
utility.preprocessing(dt, startDay=1913, finishDay=1941, savePath=test_data_path)
model.load_weights("weights_best.hdf5")

## TODo predict
predictions = utility.predict(model, startDay=1913, data_path=test_data_path, training_dataset=training_data_path)
utility.to_submission(predictions=predictions)
