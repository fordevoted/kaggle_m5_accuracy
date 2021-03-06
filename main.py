import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import utility
import keras.backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

dataPath = "./kaggle/input"
training_data_path = "./kaggle/training data"
test_data_path = "./kaggle/test data"
startDay= 350
trainFinishDay=1913
predictDay = 1941
# dt = pd.read_csv(dataPath + "/sales_train_validation.csv")
# print(dt.head(3))

# Reduce memory usage and compare with the previous one to be sure
# dt = utility.downcast_dtypes(dt)

# prepare
### utility.preprocessing(dt, startDay=startDay, finishDay=trainFinishDay, savePath=training_data_path)
# print(dt.columns)

# hyper parameter
epoch = 2000
batch_size = 32
validation_spilt = 0.2
filepath = "weights_best.hdf5"
early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=2)
checkPoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min')
# batch learning
# fit
#print("array shape  " + str(X_train.shape))
isEnd = False
startDay = utility.startDay
duration = 100
while not isEnd:
	print("startDay: " + str(startDay))
	finishDay = startDay + duration
	if finishDay > 1913:
		finishDay = 1913
		isEnd = True
	X_train, y_train = utility.prepare_training_data(startDay, finishDay, data_path=training_data_path)
	if startDay == utility.startDay:
		model = utility.create_model(X_train.shape)

	model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,
         validation_split=validation_spilt, verbose=1, callbacks=[early_stopping, checkPoint])
	startDay = finishDay

# predict
#prepare test data
# utility.preprocessing(dt, startDay=trainFinishDay, finishDay=predictDay, savePath=test_data_path)
# model.load_weights("weights_best.hdf5")

## TODo predict
# predictions = utility.predict(model, startDay=trainFinishDay, data_path=test_data_path, training_dataset=training_data_path, bias=startDay)
# utility.to_submission(predictions=predictions)
