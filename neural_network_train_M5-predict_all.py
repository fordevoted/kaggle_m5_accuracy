#!/usr/bin/env python
# coding: utf-8

# # 深度之眼 - Kaggle 比賽訓練營

# - **BaseLine步驟**：    
#                         1. 數據分析 EDA
#                         2. 特徵工程
#                         3. 模型訓練
#                         4. 線下驗證

# ## 一、數據分析
# - 查看 sales 數據前幾行
# - 查看 sales 數據聚合結果趨勢
# - 查看 sales 數據標籤分佈
'''
首先，每一條商品為一個時間序列(Sale)，
其他維度信息：商品種類，店舖ID，所在州，價格，日期，節假日信息等。

A榜  1913天數據預測後28天數據： 2016-04-25 to 2016-05-22
B榜  1941天數據預測後28天數據： 2016-05-23 to 2016-06-19

按照不同維度對時間序列進行聚合
但是可以將30490條數據按照不同維度進行聚合：
1 我們可以將30490天數據聚合成一條數據，得到1條時間序列，即level 1。
2 可以按照州的維度進行聚合，因為有3個州，所以可以聚合得到3條數據，即得到3條時間序列，即level 2.
3 可以按照商店(store)的維度進行聚合，有10家商店，所以可以聚合得到10條數據，即level 3.
4 可以按照商品類別(category)的維度進行聚合，因為有3個類目，所以可以聚合得到3條數據，即level 4。
5 可以按照產品部門(department)的維度進行聚合，因為有7個部門，所以可以局和得到7條數據，得到level 5。
6 可以按照state和category維度組合聚合，得到3*3條數據，即level 6.
7 可以按照state和department維度組合聚合，得到3*7條數據，即level7。
8 可以按照store和category維度組合聚合，得到3*10條數據，得到level8。
9 可以按照store和department組合聚合，得到10*7條數據，即level 9。
10
11
12
為什們說時間序列聚合呢？因為我們最終的目的不是預測30490條數據，而是預測42840條數據的準確度。

接下來再看一下評估指標：
WRMSSE : weighted root mean squared scaled error
相當於是MSE前面加了一個權重。WRMSSE = \sum_{i=1}^{42840}{w_i \cdot RMSSE}
其中\sum_{i=1}^{42840}{w_i} = 1。歸一化

問題：評估指標和損失函數的區別：
我們再機器學習訓練模型的時候，希望損失值越小越好；而我們再來打比賽的時候，也希望
模型的評估指標的損失越小越好。他們的區別是評估指標不一定能夠優化。從而通過損失函數替代評估指標，
如果能夠直接用評估指標進行優化/求導，那麼我們就用評估指標作為我們的損失函數，而若是不能優化，我們
就需要找一個損失函數來替代我們的評估指標，對於這個評估指標而言，損失函數可能會替代的好，也可能表現差。
若損失函數能夠替代評估指標，那麼就表明在損失函數上表現好的模型在評估指標上也會得不錯。


M5 eda得到的信息：
1 大量數據為0，且為整數
2 總體趨勢向上，，具有明顯的年週期性
3 按不同維度聚合，呈現不同趨勢
'''

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras import regularizers
import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import keras.backend as K

config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=20, inter_op_parallelism_threads=20,
                        device_count={'CPU': 20})
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

pd.set_option('display.max_colwidth', 100)

sale_data = pd.read_csv('./kaggle/input/sales_train_validation.csv')

sale_data.head(5)

day_data = sale_data[[f'd_{day}' for day in range(1, 1914)]]

# 統計商品店每一天的總銷售情況
total_sum = np.sum(day_data, axis=0).values

# 這個統計的是商品的在時間窗口內的銷售量
# total_item_sum = np.sum(day_data,axis=1).values

plt.plot(total_sum)

day_data[day_data < 100].values.reshape(-1)

# ## 二、特徵工程
#
# 選定機器學習的建模方案，核心思想是對時間序列抽取窗口特徵。

# 抽取窗口特徵：
# - 前7天
# - 前28天
# - 前7天均值
# - 前28天均值
#
# 關聯其他維度信息
#
# - 日期
# - 價格


import sys
from datetime import datetime, timedelta

'''
train_start : 指的是取多長的數據參與訓練，沒有必要讓全部的數據參與訓練。
test_start : 去多長的數據參與測試
is_train : 是訓練集呢還是測試集

這裡為什們不需要全部的樣本參與訓練呢？我個人覺得預測的銷量應該是與其相近的銷量相關的，歷史越久遠的銷售量應該是越不相關的，甚至是干擾。 
'''


def create_train_data(train_start=350, test_start=1800, is_train=True):
    # 基本參數
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16", "sell_price": "float32"}
    CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
                  "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
                  "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32'}

    start_day = train_start if is_train else test_start
    numcols = [f"d_{day}" for day in range(start_day, 1914)]
    catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    SALE_DTYPES = {numcol: "float32" for numcol in numcols}

    # dict update 方法應該是擁有修改和添加的方法
    SALE_DTYPES.update({col: "category" for col in catcols if col != "id"})

    # 加載price數據
    price_data = pd.read_csv('./kaggle/input/sell_prices.csv', dtype=PRICE_DTYPES)

    # 加載cal數據
    cal_data = pd.read_csv('./kaggle/input/calendar.csv', dtype=CAL_DTYPES)

    # 加載sale數據
    sale_data = pd.read_csv('./kaggle/input/sales_train_validation.csv', dtype=SALE_DTYPES,
                            usecols=catcols + numcols)

    # 類別標籤轉換
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            print(price_data[col].cat)
            print(price_data[col].cat.codes)

            price_data[col] = price_data[col].cat.codes.astype("int16")
            price_data[col] -= price_data[col].min()
            break

    cal_data["date"] = pd.to_datetime(cal_data["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            # 對category類型特徵進行編碼
            cal_data[col] = cal_data[col].cat.codes.astype("int16")
            cal_data[col] -= cal_data[col].min()

    for col in catcols:
        if col != "id" and col != "item_id":
            sale_data[col] = sale_data[col].cat.codes.astype("int16")
            sale_data[col] -= sale_data[col].min()

    # 注意提交格式裡有一部分為空
    if not is_train:
        for day in range(1913 + 1, 1913 + 2 * 28 + 1):
            sale_data[f"d_{day}"] = np.nan

    sale_data = pd.melt(sale_data,
                        id_vars=catcols,
                        value_vars=[col for col in sale_data.columns if col.startswith("d_")],
                        var_name="d",
                        value_name="sales")
    # 將銷量數據、價格數據和calendar數據進行關聯
    sale_data = sale_data.merge(cal_data, on="d", copy=False)
    # sale_data = sale_data.merge(price_data, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
    # sale_data = sale_data.merge(cal_data[['d', 'wm_yr_wk']], on="d", copy=False)
    sale_data = sale_data.merge(price_data, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    for col in catcols:
        if col == "item_id":
            sale_data[col] = sale_data[col].cat.codes.astype("int16")
            sale_data[col] -= sale_data[col].min()
    return sale_data


def create_feature(sale_data, is_train=True, day=None):
    # 可以在這裡加入更多的特徵抽取方法
    # 獲取7天前的數據，28天前的數據 即將7/28作為時間窗口進行特徵抽取
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]

    # 如果是測試集只需要計算一天的特徵，減少計算量
    # 注意訓練集和測試集特徵生成要一致
    if is_train:
        for lag, lag_col in zip(lags, lag_cols):
            sale_data[lag_col] = sale_data[["id", "sales"]].groupby("id")["sales"].shift(lag)
    else:
        for lag, lag_col in zip(lags, lag_cols):
            sale_data.loc[sale_data.date == day, lag_col] = sale_data.loc[
                sale_data.date == day - timedelta(days=lag), 'sales'].values

            # 將獲取7天前的數據，28天前的數據做移動平均
    wins = [7, 28]

    if is_train:
        for win in wins:
            for lag, lag_col in zip(lags, lag_cols):
                sale_data[f"rmean_{lag}_{win}"] = sale_data[["id", lag_col]].groupby("id")[lag_col].transform(
                    lambda x: x.rolling(win).mean())
    else:
        for win in wins:
            for lag in lags:
                df_window = sale_data[
                    (sale_data.date <= day - timedelta(days=lag)) & (sale_data.date > day - timedelta(days=lag + win))]
                df_window_grouped = df_window.groupby("id").agg({'sales': 'mean'}).reindex(
                    sale_data.loc[sale_data.date == day, 'id'])
                sale_data.loc[sale_data.date == day, f"rmean_{lag}_{win}"] = df_window_grouped.sales.values

                # 處理時間特徵
    # 有的時間特徵沒有，通過datetime的方法自動生成
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in sale_data.columns:
            sale_data[date_feat_name] = sale_data[date_feat_name].astype("int16")
        else:
            sale_data[date_feat_name] = getattr(sale_data["date"].dt, date_feat_func).astype("int16")
    return sale_data


def rmse(y_true, y_pred):
    diff = tf.keras.backend.clip(y_pred, - sys.float_info.max, sys.float_info.max) - y_true
    square_val = tf.keras.backend.clip(K.square(diff), - sys.float_info.max, sys.float_info.max)
    mean_val = K.mean(square_val, axis=-1)
    sqrt_val = K.sqrt(mean_val)
    return sqrt_val


def regression_model(dropout=5.0e-5, learning_rate=5.0e-5, l2_val=5.0e-5):
    optimizer = Adam(lr=learning_rate, beta_1=0.999, beta_2=0.999, amsgrad=False)
    model = keras.Sequential()
    model.add(Dense(128, activation='elu', input_shape=(25,)))
    model.add(Dense(64, activation='elu', kernel_regularizer=regularizers.l2(l2_val),
                    bias_regularizer=regularizers.l2(l2_val)))
    model.add(Dense(16, activation='elu', kernel_regularizer=regularizers.l2(l2_val),
                    bias_regularizer=regularizers.l2(l2_val)))
    model.add(Dense(32, activation='elu', kernel_regularizer=regularizers.l2(l2_val),
                    bias_regularizer=regularizers.l2(l2_val)))
    model.add(Dense(8, activation='elu', kernel_regularizer=regularizers.l2(l2_val),
                    bias_regularizer=regularizers.l2(l2_val)))
    model.add(Dense(8, activation='elu', kernel_regularizer=regularizers.l2(l2_val),
                    bias_regularizer=regularizers.l2(l2_val)))
    model.add(Dense(4, activation='elu', kernel_regularizer=regularizers.l2(l2_val),
                    bias_regularizer=regularizers.l2(l2_val)))
    model.add(Dense(1, activation='relu', activity_regularizer=regularizers.l2(l2_val)))
    model.add(Dropout(dropout))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
    return model


sale_data = create_train_data(train_start=350, is_train=True)
# sys.exit()
sale_data = create_feature(sale_data)

# 清洗數據，選擇需要訓練的數據
sale_data.dropna(inplace=True)
cat_feats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1",
                                                                        "event_type_2"]
useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
train_cols = sale_data.columns[~sale_data.columns.isin(useless_cols)]
X_train = sale_data[train_cols]
y_train = sale_data["sales"]
X_train = X_train[: int(len(X_train) / 100)]
X_train.to_csv("training_data.csv", index = False)
print(X_train.head(5))
print(y_train.head(5))
exit(0)

# ## 三、模型訓練
#
# 選擇 LGB 模型進行模型的訓練。
'''

# - 損失函數的選擇
# - 預測時候的技巧

這裡的損失函數為什們選擇tweedie損失呢？而不是MSE損失呢？
主要是因為標籤的的分佈符合泊松分佈/tweedie分佈，從而不能選擇MSE損失，因為MSE是高斯分佈。

# tweedie_variance_power 參數的選擇 [1,2] 之間。
# tweedie分佈中有一個參數p，當 p=1的時候，tweedie分佈r就是泊松分佈，當p=2時，tweedie分佈為Gama分佈。
# 在1和2之間，就靠近誰，就更像誰。


# LGB 模型是 GBDT 模型的變種，無法突然訓練集的上界。
因為LGB是對空間的劃分，無法突破原有空間內的上限值，而我們預測的銷售額都是呈現上升趨勢，

改進方案1：
那麼就需要讓LGB打破這個趨勢，就通過在LGB前添加參數，如1.1倍的參數來進行擬合；或者是更大的參數進行擬合；這樣就能夠得到上升到趨勢，
改進方案2：
將LGB的預測結果運用到時間序列模型，如趨勢擬合profit模型等。

'''


def predict_ensemble_for_regression(train_cols, m_regression):
    date = datetime(2016, 4, 25)
    # alphas = [1.035, 1.03, 1.025, 1.02]
    # alphas = [1.028, 1.023, 1.018]
    alphas = [1.035, 1.03, 1.025]  # 這裡用改進方案1進行預測，
    weights = [1 / len(alphas)] * len(alphas)
    sub = 0.

    test_data = create_train_data(is_train=False)

    '''
    預測過程：用第一天的預測結果，經過特徵工程加工後，在預測第二天的數值；對二天的結果進行特徵工程加工，預測得到第三天的數值。
    循環迭代往前預測的過程。
    '''
    for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

        test_data_c = test_data.copy()
        cols = [f"F{i}" for i in range(1, 29)]

        for i in range(0, 28):
            day = date + timedelta(days=i)
            print(i, day)
            tst = test_data_c[(test_data_c.date >= day - timedelta(days=57)) & (test_data_c.date <= day)].copy()
            tst = create_feature(tst, is_train=False, day=day)
            tst = tst.loc[tst.date == day, train_cols]

            tst_ndarray = tst.to_numpy()
            # test_data_c.loc[test_data_c.date == day, "sales"] = alpha * m_regression.predict(tst)
            test_data_c.loc[test_data_c.date == day, "sales"] = alpha * m_regression.predict(tst_ndarray)

        # 改為提交數據的格式
        test_sub = test_data_c.loc[test_data_c.date >= date, ["id", "sales"]].copy()
        test_sub["F"] = [f"F{rank}" for rank in test_sub.groupby("id")["id"].cumcount() + 1]
        test_sub = test_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
        test_sub.fillna(0., inplace=True)
        test_sub.sort_values("id", inplace=True)
        test_sub.reset_index(drop=True, inplace=True)
        test_sub.to_csv(f"submission_{icount}.csv", index=False)
        if icount == 0:
            sub = test_sub
            sub[cols] *= weight
        else:
            sub[cols] += test_sub[cols] * weight
        print(icount, alpha, weight)

    sub2 = sub.copy()
    # 把大於28天後的validation替換成evaluation
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    sub.to_csv("submissionV3_dnn_regression.csv", index=False)


num_epochs = 30
batch_size = 30

X_train_ndarray = X_train.to_numpy()
y_train_ndarray = y_train.to_numpy()
del X_train, y_train
split_x_train, split_x_val, split_y_train, split_y_val = train_test_split(X_train_ndarray, y_train_ndarray,
                                                                          test_size=0.2,
                                                                          random_state=2)

regression = KerasRegressor(build_fn=regression_model,
                            batch_size=batch_size, epochs=num_epochs)
regression.fit(split_x_train, split_y_train, verbose=2, validation_data=(split_x_val, split_y_val))

predict_ensemble_for_regression(train_cols, regression)
