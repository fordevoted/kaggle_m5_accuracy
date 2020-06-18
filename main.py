import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import utility

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dataPath = "./kaggle/input"

dt = pd.read_csv(dataPath + "/sales_train_validation.csv")
dt.head(3)
# Reduce memory usage and compare with the previous one to be sure
dt = utility.downcast_dtypes(dt)
dt, daysBeforeEventTest, event_typeTest, week_dayTest = utility.preprocessing(dt)
print(dt.columns)

X_train, y_train = utility.prepare_training_data(dt)
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
model.load_weights("weights_best.hdf5")
predictions = utility.predict(model, dt, daysBeforeEventTest, event_typeTest, week_dayTest)
utility.to_submission(predictions=predictions)
