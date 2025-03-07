# coding=UTF-8
import numpy as np

# from tensorflow import keras
# from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Input, LeakyReLU

from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Input, LeakyReLU

from tensorflow.keras.models import Sequential


from tcn.tcn import TCN

class modelss:
    def __init__(self,X_train, X_test, Y_train, Y_test, scaled_tool):
        '''

        Parameters
        ----------
        self.epochs  = 60  定义了模型训练的轮数，即模型将完整地遍历训练数据集的次数。
        self.batch   = 32
        指定了训练过程中的批量大小，即每次从训练数据集中选取多少个样本进行参数更新。
        较小的批量大小可能会使模型训练过程更加随机，有助于跳出局部最小值，但可能会增加训练时间；较大的批量大小可能会使训练过程更稳定，但可能需要更多的内存资源。
        self.units   = 20
        设置了模型中某些层（可能是如 GRU 层等）的单元数量，单元数量决定了模型能够学习到的特征表示的复杂程度

        X_train = data_input[:-test_number]# (2658,20)
        Y_train = data_label[:-test_number]# (2685,1)
        X_test = data_input[-test_number:]# (200,20)
        Y_test = data_label[-test_number:]# (200,1)

        '''
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_tedst = Y_test
        self.scaled  = scaled_tool
        self.epochs  = 60
        self.batch   = 28
        self.units   = 20

    def run_tcn_gru(self):
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))  # 2658 20 1
        X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1)) # 200 20 1
        ####搭建预测模型
        model = Sequential() #  Keras 中构建神经网络模型的一种简单方式，允许按顺序堆叠层
        model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))# (None,20,1) batch_shape 中的 None 表示批次大小可以动态调整  步长  通道数
        model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1, 2, 4], return_sequences=True))
        # model.add(LSTM(units=self.units, return_sequences=True))
        model.add(GRU(units=self.units, return_sequences= False))
        model.add(Dense(20)) # 添加有 10 个神经元的全连接层
        model.add(LeakyReLU(alpha=0.3)) # alpha 为 0.3 的 LeakyReLU 激活函数层
        model.add(Dense(1))  # 输出单个值的全连接层
        ###配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae') # 配置模型的学习过程，使用Adam优化器和均方误差损失函数，评价指标为平均绝对误差（MAE）。
        model.summary() # 打印模型的概述，包括每层的参数数量。
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # X_test 的形状是 (200, 20, 1)，即 200 个样本，每个样本包含 20 个时间步长，单通道数据 。
        # 模型最后一层是 Dense(1)，这意味着对于每个输入样本，模型输出一个标量值。
        Y_pre = model.predict(X_test)
        print('model.predict(X_test)   : ==============> ',Y_pre.shape)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_tcn_lstm(self):
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        ####搭建预测模型
        model = Sequential()
        model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))
        model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1, 2, 4], return_sequences=True))
        model.add(LSTM(units=self.units, return_sequences= False))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1))
        ###配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_tcn_rnn(self):
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        ####搭建预测模型
        model = Sequential()
        model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))
        model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1,2,4],return_sequences=True))
        model.add(SimpleRNN(units=self.units, return_sequences= False))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1))
        ###配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_tcn_bpnn(self):
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        ####搭建预测模型
        model = Sequential()
        model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))
        model.add(TCN(nb_filters=10, kernel_size=2, dilations=[1, 2, 4], return_sequences=False))
        model.add(Dense(self.units))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(self.units))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(self.units))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1))
        ###配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre


    def run_GRU(self):
        # 张量转化
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        # 搭建预测模型
        model = Sequential()
        model.add(GRU(units=self.units, input_shape=(X_train.shape[1],1)))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_LSTM(self):
        # 张量转化
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        # 搭建预测模型
        model = Sequential()
        model.add(LSTM(units=self.units, input_shape=(X_train.shape[1],1)))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre

    def run_RNN(self):
        # 张量转化
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        # 搭建预测模型
        model = Sequential()
        model.add(SimpleRNN(units=self.units, input_shape=(X_train.shape[1],1)))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre
    def run_BPNN(self):
        model = Sequential()
        model.add(Dense(self.units, activation='relu', input_shape=(self.X_train.shape[1],)))
        model.add(Dense(self.units, activation='relu'))
        model.add(Dense(self.units, activation='relu'))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics='mae')
        model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(self.X_test)

        Y_pre = Y_pre.reshape(self.X_test.shape[0], 1)
        Y_pre = self.scaled.inverse_transform(Y_pre)
        return Y_pre