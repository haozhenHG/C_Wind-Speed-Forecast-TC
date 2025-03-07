#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/12/27 18:43
# @File ：test.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
import numpy as np
from keras.models import Sequential
from keras.layers import Input, TCN, GRU, Dense, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
# 假设X_train, X_test, Y_train, Y_test已经存在且数据格式正确
# 这里只是为了示例完整性，实际需替换为真实数据
X_train = np.random.rand(2658, 20)
Y_train = np.random.rand(2658, 1)
X_test = np.random.rand(200, 20)
Y_test = np.random.rand(200, 1)

scaled_tool = MinMaxScaler()
scaled_tool.fit(Y_train)
Y_train = scaled_tool.transform(Y_train)


param_grid = {
    'epochs': [30, 60, 90],
    'batch': [16, 32, 64],
    'units': [10, 20, 30]
}


best_loss = float('inf')
best_params = None

for epochs in param_grid['epochs']:
    for batch in param_grid['batch']:
        for units in param_grid['units']:
            model = Sequential()
            X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            model.add(Input(batch_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
            model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1, 2, 4], return_sequences=True))
            model.add(GRU(units=units, return_sequences=False))
            model.add(Dense(10))
            model.add(LeakyReLU(alpha=0.3))
            model.add(Dense(1))

            model.compile(optimizer='Adam', loss='mse', metrics='mae')
            model.fit(X_train_reshaped, Y_train, epochs=epochs, batch_size=batch, verbose=0)

            loss = model.evaluate(X_test_reshaped, scaled_tool.transform(Y_test), verbose=0)[0]
            if loss < best_loss:
                best_loss = loss
                best_params = {'epochs': epochs, 'batch': batch, 'units': units}

print("Best parameters: ", best_params)
print("Best loss: ", best_loss)