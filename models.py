# coding=UTF-8
import numpy as np
from keras.callbacks import EarlyStopping
from tensorflow.compiler.mlir import tensorflow

# from tensorflow import keras
# from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Input, LeakyReLU

from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Input, LeakyReLU, Dropout
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tcn.tcn import TCN

class modelss:
    def __init__(self,X_train, X_test, Y_train, Y_test,  scaler):
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
        self.scaler  = scaler
        self.epochs  = 128
        self.batch   = 128
        self.units   = 20

    # def run_tcn_gru(self):
    #     # 数据重塑
    #     # X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))  # 2658 20 1
    #     # X_test  = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1)) # 200 20 1
    #     X_train = self.X_train
    #     X_test = self.X_test
    #     ####搭建预测模型
    #     model = Sequential() #  Keras 中构建神经网络模型的一种简单方式，允许按顺序堆叠层
    #     model.add(Input(batch_shape=(None, X_train.shape[1], X_train.shape[2])))# (None,20,1) batch_shape 中的 None 表示批次大小可以动态调整  步长  通道数
    #     model.add(TCN(nb_filters=64, kernel_size=3, dilations=[1,2,4,8], return_sequences=True,dropout_rate=0.2))
    #     # model.add(LSTM(units=self.units, return_sequences=True))
    #     # model.add(GRU(units=self.units, return_sequences= False))
    #     # model.add(Dense(10)) # 添加有 10 个神经元的全连接层
    #     # model.add(LeakyReLU(alpha=0.3)) # alpha 为 0.3 的 LeakyReLU 激活函数层
    #     model.add(GRU(units=128, return_sequences=True))  # 第一层 GRU
    #     model.add(Dropout(0.3))  # 添加 Dropout  过 Dropout 缓解过拟合（时间序列模型容易过拟合）
    #     model.add(GRU(units=64, return_sequences=False))  # 第二层 GRU
    #     # model.add(Dense(1))  # 输出单个值的全连接层
    #     model.add(Dense(32, activation="swish"))  # 替换 LeakyReLU 为 Swish 激活函数
    #     model.add(Dense(1))  # 输出层无需激活函数（回归任务）
    #
    #     ###配置和训练
    #     model.compile(optimizer='Adam', loss='mse', metrics=['mae']) # 配置模型的学习过程，使用Adam优化器和均方误差损失函数，评价指标为平均绝对误差（MAE）。
    #     model.summary() # 打印模型的概述，包括每层的参数数量。
    #     lr_scheduler = ReduceLROnPlateau(
    #         monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
    #     )
    #     model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch,validation_split=0.1,  callbacks=[EarlyStopping(monitor='val_loss', patience=10),lr_scheduler])
    #     # X_test 的形状是 (200, 20, 1)，即 200 个样本，每个样本包含 20 个时间步长，单通道数据 。
    #     # 模型最后一层是 Dense(1)，这意味着对于每个输入样本，模型输出一个标量值。
    #     Y_pre = model.predict(X_test)
    #     print('model.predict(X_test)   : ==============> ',Y_pre.shape)
    #
    #     Y_pre = Y_pre.reshape(X_test.shape[0], 1)
    #     Y_pre = self.label_scaled_tool.inverse_transform(Y_pre)
    #     return Y_pre

    def run_tcn_gru(self):
        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))

        # 增强TCN层
        model.add(TCN(
            nb_filters=128,  # 增加滤波器数量
            kernel_size=4,  # 扩大卷积核
            dilations=[1, 2, 4, 8, 16],  # 扩大感受野
            return_sequences=True,
            dropout_rate=0.3,  # 提升Dropout比率

        ))

        # 改进GRU层结构
        model.add(GRU(
            units=256,
            return_sequences=True,  # 保持时间步信息
            kernel_regularizer=l2(1e-4)
        ))
        model.add(Dropout(0.4))
        model.add(GRU(
            units=128,
            return_sequences=False,
            kernel_regularizer=l2(1e-4)
        ))

        # 更深的全连接层
        model.add(Dense(64, activation="swish", kernel_regularizer=l2(1e-4)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation="swish"))
        model.add(Dense(1))

        # 优化训练策略
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # 添加早停和动态学习率
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]

        model.fit(
            self.X_train, self.Y_train,
            epochs=100,  # 增加训练轮数
            batch_size=128,  # 增大批量大小
            validation_split=0.2,  # 更多验证数据
            callbacks=callbacks
        )

        Y_pre = model.predict(self.X_test)
        #  保持特征-标签维度一致
        #  确保在调用 inverse_transform 时输入的数据形状和 fit_transform 时的数据形状一致。
        # 在进行反归一化之前，将预测结果和特征合并成合适的形状
        Y_pre_actual = self.scaler.inverse_transform(np.concatenate([self.X_test[:, -1, :], Y_pre], axis=1) )[:, -1]
        return Y_pre_actual

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
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.label_scaled_tool.inverse_transform(Y_pre)
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
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.label_scaled_tool.inverse_transform(Y_pre)
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
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
        model.summary()
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.label_scaled_tool.inverse_transform(Y_pre)
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
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.label_scaled_tool.inverse_transform(Y_pre)
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
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.label_scaled_tool.inverse_transform(Y_pre)
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
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
        model.fit(X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(X_test)

        Y_pre = Y_pre.reshape(X_test.shape[0], 1)
        Y_pre = self.label_scaled_tool.inverse_transform(Y_pre)
        return Y_pre
    def run_BPNN(self):
        model = Sequential()
        model.add(Dense(self.units, activation='relu', input_shape=(self.X_train.shape[1],)))
        model.add(Dense(self.units, activation='relu'))
        model.add(Dense(self.units, activation='relu'))
        model.add(Dense(1))
        # 配置和训练
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
        model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch)
        # model.summary()
        # model.predict()对模型进行预测
        Y_pre = model.predict(self.X_test)

        Y_pre = Y_pre.reshape(self.X_test.shape[0], 1)
        Y_pre = self.label_scaled_tool.inverse_transform(Y_pre)
        return Y_pre