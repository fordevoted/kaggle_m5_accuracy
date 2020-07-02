# kaggle_m5_accuracy
kaggle competition host by University of Nicosia<br>
https://www.kaggle.com/c/m5-forecasting-accuracy

## Reference
[1] https://github.com/hitottiez/cnn_lstm/blob/master/run.py<br>
[2] https://www.kaggle.com/bountyhunters/baseline-lstm-with-keras-0-7
 

## Introduction 
data contains goods info include sales in alse_train_validation.csv, event and info related to days, goods prices are recorded in sell_prices.csv by date.So in this task, I try to view it as video caption task, which means input will be few of images(2D tensor), then output the goods sales of one day.<br>
each input contain 14 image, which means 14 days data, and predict the 15th day ssales. data using start from day 350[2]<br>
the missing value usually using mean value to complement. 

## Model
refer reference[1] model, which contain CNN and LSTM.

## Perfomance
without any hyperparameter tuing, 1.0213 on kagge leaderboard

## Contact
210509fssh@gmail.com
