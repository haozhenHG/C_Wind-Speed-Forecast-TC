# coding=UTF-8
'''
@Date    : 2022.05.29
@Author  : Jethro
'''
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models import modelss
from Decomposition import Decomposition
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import warnings
warnings.filterwarnings('ignore')
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def denoise(data, imfs):
    '''
    目的是逐步累加本征模态函数（IMFs），并在每次累加后检查累加结果与原始数据的相关性，一旦相关性达到设定阈值（0.995），就停止累加。
    这种做法常用于信号处理或数据分析中的降噪场景，通过选取合适的 IMFs 组合来重构一个与原始信号尽可能相似的 “干净” 信号
    Parameters
    ----------
    data     源数据中的某一列数据
    imfs     ssa_imfs  = Decomp.SSA()中的 ssa_imfs

    Returns
    -------

    '''
    data = data.reshape(-1) # 将多维数组展平为一维数组
    print('denoise ',data.shape)  # denoise  (2880,)
    denoise_data = 0
    for i in range(imfs.shape[1]):  # imfs.shape (2880,15)
        denoise_data += imfs[:,i]
        pearson_corr_coef = np.corrcoef(denoise_data, data)
        if pearson_corr_coef[0,1] >=0.995:
            print(i) # 8  前9列数据
            break

    return denoise_data

def IMF_decomposition(data, length):
    '''

    Parameters
    ----------
    data  : 源数据中的某一列数据     代码中选择的是第三列数据
    length ： 同imfs_number 数值 15
    Returns
    -------
    '''
    Decomp = Decomposition(data, length)  # 初始化对象
    # emd_imfs  = Decomp.EMD()
    # eemd_imfs = Decomp.EEMD()
    # vmd_imfs  = Decomp.VMD()
    ssa_imfs  = Decomp.SSA()  # 调用方法奇异谱分析方法
    print('IMF_decomposition:',ssa_imfs.shape)
    # 目标文件路径
    # file_path = '/De_data/ssa_imfs_6.csv'
    # # 检查目录是否存在，如果不存在则创建
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # np.savetxt(r'De_data/ssa_imfs_6.csv',ssa_imfs,delimiter = ',')

    # emd_denoise = denoise(data, emd_imfs)
    # eemd_denoise = denoise(data, eemd_imfs)
    # vmd_denoise = denoise(data, vmd_imfs)
    ssa_denoise = denoise(data, ssa_imfs)
    # emd_denoise, eemd_denoise, vmd_denoise,

    return ssa_denoise

# def Data_partitioning(data,label,test_number, input_step, pre_step):
#     '''
#     data,test_number, input_step, pre_step = IMFS重构后干净数据   200  20  2
#
#     Parameters
#     ----------
#     input_step  定义输入数据的时间步长，也就是在构建预测模型时，每个样本所包含的历史数据的长度
#     Returns
#     -------
#     '''
#     # 分离特征和标签
#     features = data.values[:, :-1]  # 前 10 列作为特征
#     labels = label.reshape(-1, 1)  # 最后一列作为标签
#
#     test_number = test_number # 200
#     # 对特征和标签进行归一化
#     feature_scaled_tool = MinMaxScaler(feature_range=(0, 1)) # 元组 并非列表
#     features_scaled = feature_scaled_tool.fit_transform(features)
#
#     label_scaled_tool = MinMaxScaler(feature_range=(0, 1))
#     labels_scaled = label_scaled_tool.fit_transform(labels)
#
#     # 切片
#     step_size = input_step   # 输入步长 窗口 168
#     num_samples = features_scaled.shape[0]
#     num_features = features_scaled.shape[1]
#     data_input = np.zeros((num_samples - step_size - pre_step, step_size, num_features))
#     data_label = np.zeros((num_samples - step_size - pre_step, 1))
#
#     for i in range(num_samples - step_size - pre_step):
#         data_input[i, :, :] = features_scaled[i:i + step_size, :]
#         data_label[i, 0] = labels_scaled[step_size + i + pre_step, 0]
#
#     print(f'data_input.shape is {data_input.shape}, data_label.shape is {data_label.shape}')
#     '''
#     for 0 in (2858):
#         data_input[0,:] = data_scaled[0:20,0]
#         data_label[0,0] = data_scaled[22,0]
#
#     for 2857 in (2858):
#         data_input[2857,:] = data_scaled[2857:2879,0]
#         data_label[2857,0] = data_scaled[2879,0]
#     '''
#     # data_label = data_scaled[step_size+1:,0]
#     # 划分数据集
#     X_train = data_input[:-test_number]# (2658,20)
#     Y_train = data_label[:-test_number]# (2685,1)
#     X_test = data_input[-test_number:]# (200,20)
#     Y_test = data_label[-test_number:]# (200,1)
#
#     return  X_train, X_test, Y_train, Y_test,  feature_scaled_tool, label_scaled_tool
def Data_partitioning(data, label, test_number, input_step, pre_step):
    # 检查数据和标签长度是否一致
    if len(data) != len(label):
        raise ValueError("数据和标签的长度必须一致。")

    # 按时间顺序划分训练集和测试集
    split_idx = len(data) - test_number
    train_data = data[:split_idx]
    train_label = label[:split_idx]
    test_data = data[split_idx - input_step - pre_step:]
    test_label = label[split_idx - input_step - pre_step:]

    # 合并训练集的特征和标签后归一化
    combined_train = np.concatenate([train_data, train_label.reshape(-1, 1)], axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_train_scaled = scaler.fit_transform(combined_train)

    # 合并测试集的特征和标签后进行归一化（使用训练集的缩放器）
    combined_test = np.concatenate([test_data, test_label.reshape(-1, 1)], axis=1)
    combined_test_scaled = scaler.transform(combined_test)

    # 重新分离训练集的特征和标签
    train_features_scaled = combined_train_scaled[:, :-1]
    train_labels_scaled = combined_train_scaled[:, -1].reshape(-1, 1)

    # 重新分离测试集的特征和标签
    test_features_scaled = combined_test_scaled[:, :-1]
    test_labels_scaled = combined_test_scaled[:, -1].reshape(-1, 1)

    # 为训练集生成时间窗口
    def create_dataset(features, labels, input_step, pre_step):
        num_samples = features.shape[0]
        data_input = []
        data_label = []
        for i in range(num_samples - input_step - pre_step):
            data_input.append(features[i:i + input_step, :])
            data_label.append(labels[i + input_step + pre_step, 0])
        return np.array(data_input), np.array(data_label).reshape(-1, 1)

    X_train, Y_train = create_dataset(train_features_scaled, train_labels_scaled, input_step, pre_step)
    X_test, Y_test = create_dataset(test_features_scaled, test_labels_scaled, input_step, pre_step)

    return X_train, X_test, Y_train, Y_test, scaler

def single_model(data,label,test_number,flag, input_step, pre_step):
    '''
    test_number, imfs_number, input_step, pre_step= 200, 15, 20, 2
    Parameters
    ----------
    data        使用IMFS函数重构成干净数据  (2880,0)
    flag        使用的模型
    Returns
    -------

    '''
    X_train, X_test, Y_train, Y_test, scaler = Data_partitioning(data,label,test_number, input_step, pre_step) # 切片
    print('*'*30)
    print('\tData_partitioning\t')
    print('*' * 30)
    print(f'X_train.shape is {X_train.shape}, X_test.shape is {X_test.shape}\nY_train.shape is {Y_train.shape}, Y_test.shape is {Y_test.shape}')

    model = modelss(X_train, X_test, Y_train, Y_test,  scaler)
    if flag == 'tcn_gru':
        pre = model.run_tcn_gru()
    if flag == 'tcn_lstm':
        pre = model.run_tcn_lstm()
    if flag == 'tcn_rnn':
        pre = model.run_tcn_rnn()
    if flag == 'tcn_bpnn':
        pre = model.run_tcn_bpnn()
    if flag == 'gru':
        pre = model.run_GRU()
    if flag == 'lstm':
        pre = model.run_LSTM()
    if flag == 'rnn':
        pre = model.run_RNN()
    if flag == 'bpnn':
        pre = model.run_BPNN()
    # data_pre = pre[:, 0]

    # return data_pre

    return pre

if __name__ == '__main__':
    # test_number 测试集  input_step 输入步长 24*7
    test_number, imfs_number, input_step, pre_step= 720, 6, 168, 1  # input_step的大小对IMF_S的 mape没有影响

    # data = pd.read_csv('10 min wind speed data.csv', header= None) # (2880,3)
    # print(data.shape,data.iloc[:,2].values.shape)
    # ssa_denoise = IMF_decomposition(data.iloc[:,2].values, imfs_number)  # 选取第三列数据  经过降噪之后
    # np.savetxt('ssa_denoise_3.csv',ssa_denoise[-test_number:],delimiter = ',') # 使用IMFS函数重构成干净数据 ssa_denoise [2880]

    data = pd.read_csv('jupyter/American_final_to_model.csv', index_col=0)  # (21769, 11)
    print('输入模型数据大小：',data.shape)
    # print(data.iloc[:, -1].values) # 查看负荷列数据
    ssa_denoise_load = IMF_decomposition(data.iloc[:, -1].values, imfs_number)  # 选取负荷列数据  经过降噪之后
    np.savetxt(r'De_data/ssa_denoise_load_720.csv', ssa_denoise_load[-test_number:], delimiter=',')

    # pre_emd = single_model(emd_denoise, test_number, 'tcn_gru', input_step, pre_step)
    # pre_eemd = single_model(eemd_denoise, test_number, 'tcn_gru', input_step, pre_step)
    # pre_vmd = single_model(vmd_denoise, test_number, 'tcn_gru', input_step, pre_step)

    pre_ssa_tcn_gru = single_model(data,ssa_denoise_load, test_number, 'tcn_gru', input_step, pre_step)
    # pre_ssa_tcn_lstm = single_model(ssa_denoise, test_number, 'tcn_lstm', input_step, pre_step)
    # pre_ssa_tcn_rnn = single_model(ssa_denoise, test_number, 'tcn_rnn', input_step, pre_step)
    # pre_ssa_tcn_bpnn = single_model(ssa_denoise, test_number, 'tcn_bpnn', input_step, pre_step)
    # pre_ssa_gru = single_model(ssa_denoise, test_number, 'gru', input_step, pre_step)
    # pre_ssa_lstm = single_model(ssa_denoise, test_number, 'lstm', input_step, pre_step)
    # pre_ssa_rnn = single_model(ssa_denoise, test_number, 'rnn', input_step, pre_step)
    # pre_ssa_bpnn = single_model(ssa_denoise, test_number, 'bpnn', input_step, pre_step)

    # np.savetxt('pre_emd.csv', pre_emd, delimiter=',')
    # np.savetxt('pre_eemd.csv', pre_eemd, delimiter=',')
    # np.savetxt('pre_vmd.csv', pre_vmd, delimiter=',')

    np.savetxt('De_data/pre_ssa_tcn_gru.csv', pre_ssa_tcn_gru, delimiter=',')
    # np.savetxt('pre_ssa_tcn_lstm.csv', pre_ssa_tcn_lstm, delimiter=',')
    # np.savetxt('pre_ssa_tcn_rnn.csv', pre_ssa_tcn_rnn, delimiter=',')
    # np.savetxt('pre_ssa_tcn_bpnn.csv', pre_ssa_tcn_bpnn, delimiter=',')
    # np.savetxt('pre_ssa_gru.csv', pre_ssa_gru, delimiter=',')
    # np.savetxt('pre_ssa_lstm.csv', pre_ssa_lstm, delimiter=',')
    # np.savetxt('pre_ssa_rnn.csv', pre_ssa_rnn, delimiter=',')
    # np.savetxt('pre_ssa_bpnn.csv', pre_ssa_bpnn, delimiter=',')
    Actual = ssa_denoise_load[-test_number:]    # ssa 后的 后两百条数据

    print('实验一：不同分解方法对预测结果影响')
    '''
    数值都是越小越好
    MAE 是预测值与真实值之差的绝对值的平均值 
    MAPE 是预测值与真实值相对误差（百分比形式）的绝对值的平均值
    RMSE 是预测值与真实值之差的平方的平均值的平方根
    '''
    # print('#########################')
    # print('EMD分解方法： MAE : ', mae(Actual, pre_emd))
    # print('EMD分解方法： R2 : ', mape(Actual, pre_emd))
    # print('EMD分解方法： RMSE : ', np.sqrt(mse(Actual, pre_emd)))
    # print('#########################')
    # print('EEMD分解方法： MAE : ', mae(Actual, pre_eemd))
    # print('EEMD分解方法： R2 : ', mape(Actual, pre_eemd))
    # print('EEMD分解方法： RMSE : ', np.sqrt(mse(Actual, pre_eemd)))
    # print('#########################')
    # print('VMD分解方法： MAE : ', mae(Actual, pre_vmd))
    # print('VMD分解方法： R2 : ', mape(Actual, pre_vmd))
    # print('VMD分解方法： RMSE : ', np.sqrt(mse(Actual, pre_vmd)))
    # print('#########################')
    print('SSA分解方法： MAE : ', mae(Actual, pre_ssa_tcn_gru))
    print('SSA分解方法： MAPE : ', mape(Actual, pre_ssa_tcn_gru))
    print('SSA分解方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_gru)))

    time.sleep(5)  # 等待程序结束
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")  # 关闭显示器（需管理员权限）

    # print('实验二：采用不同时间信息提取模型对实验结果影响')
    # print('#########################')
    # print('TCN-LSTM方法： MAE : ', mae(Actual, pre_ssa_tcn_lstm))
    # print('TCN-LSTM方法： R2 : ', mape(Actual, pre_ssa_tcn_lstm))
    # print('TCN-LSTM方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_lstm)))
    # print('#########################')
    # print('TCN-RNN方法： MAE : ', mae(Actual, pre_ssa_tcn_rnn))
    # print('TCN-RNN方法： R2 : ', mape(Actual, pre_ssa_tcn_rnn))
    # print('TCN-RNN方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_rnn)))
    # print('#########################')
    # print('TCN-BPNN方法： MAE : ', mae(Actual, pre_ssa_tcn_bpnn))
    # print('TCN-BPNN方法： R2 : ', mape(Actual, pre_ssa_tcn_bpnn))
    # print('TCN-BPNN方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_bpnn)))
    #
    # print('实验三：与基模性对比')
    # print('#########################')
    # print('GRU方法： MAE : ', mae(Actual, pre_ssa_gru))
    # print('GRU方法方法： R2 : ', mape(Actual, pre_ssa_tcn_lstm))
    # print('GRU方法方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_tcn_lstm)))
    # print('#########################')
    # print('LSTM方法： MAE : ', mae(Actual, pre_ssa_lstm))
    # print('LSTM方法： R2 : ', mape(Actual, pre_ssa_lstm))
    # print('LSTM方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_lstm)))
    # print('#########################')
    # print('RNN方法： MAE : ', mae(Actual, pre_ssa_rnn))
    # print('RNN方法： R2 : ', mape(Actual, pre_ssa_rnn))
    # print('RNN方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_rnn)))
    # print('#########################')
    # print('BPNN方法： MAE : ', mae(Actual, pre_ssa_bpnn))
    # print('BPNN方法： R2 : ', mape(Actual, pre_ssa_bpnn))
    # print('BPNN方法： RMSE : ', np.sqrt(mse(Actual, pre_ssa_bpnn)))
    #
    #
    # plt.figure(2)
    # plt.plot(pre_ssa_tcn_gru, color = 'black', label= 'pre_ssa_tcn_gru')
    # plt.plot(pre_ssa_tcn_lstm, color= 'm', label= 'pre_ssa_tcn_lstm')
    # plt.plot(pre_ssa_tcn_rnn, color= 'y', label= 'pre_ssa_tcn_rnn')
    # plt.plot(pre_ssa_tcn_bpnn, color = 'red', label= 'pre_ssa_tcn_bpnn')
    # plt.plot(pre_ssa_gru, color= 'y', label= 'pre_ssa_gru')
    # plt.plot(pre_ssa_lstm, color = 'red', label= 'pre_ssa_lstm')
    # plt.plot(Actual, color= 'blue', label= 'Actual')
    # plt.legend()
    # plt.show()
