# coding=UTF-8
'''
@Date    : 2022.05.29
@Author  : Jethro
'''
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
    data = data.reshape(-1)
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
    vmd_imfs  = Decomp.VMD()
    ssa_imfs  = Decomp.SSA()  # 调用方法奇异谱分析方法
    print(ssa_imfs.shape)  ## (2880, 15)
    # np.savetxt('ssa_imfs.csv',ssa_imfs,delimiter = ',')

    # emd_denoise = denoise(data, emd_imfs)
    # eemd_denoise = denoise(data, eemd_imfs)
    vmd_denoise = denoise(data, vmd_imfs)
    ssa_denoise = denoise(data, ssa_imfs)
    # emd_denoise, eemd_denoise, vmd_denoise,

    return ssa_denoise

def Data_partitioning(data,test_number, input_step, pre_step):
    '''
    data,test_number, input_step, pre_step = IMFS重构后干净数据   200  20  2

    Parameters
    ----------
    input_step  定义输入数据的时间步长，也就是在构建预测模型时，每个样本所包含的历史数据的长度
    Returns
    -------
    '''
    # 导入数据
    dataset = data.reshape(-1,1)  # 将一维的data重塑为列向量形式（2880，1）
    test_number = test_number # 200
    # #归一化   dataset 中的元素被归一化到 0 到 1 之间
    scaled_tool = MinMaxScaler(feature_range=(0, 1)) # 最大最小归一化
    data_scaled = scaled_tool.fit_transform(dataset)  # (2880,1)
    # 切片
    step_size = input_step # 20
    data_input= np.zeros((len(data_scaled) - step_size - pre_step, step_size))  # 2858，20
    data_label = np.zeros((len(data_scaled) - step_size - pre_step, 1)) # 2858，1
    for i in range(len(data_scaled) - step_size-pre_step):  # range(2858)   0- 2857
        # 2857
        data_input[i, :] = data_scaled[i:step_size + i,0]  # 0 限定了 第一列  只有一列数据  data_scaled[0:20,0]
        data_label[i, 0] = data_scaled[step_size + i + pre_step,0] # data_scaled[20+0+2,0]
    '''
    for 0 in (2858):
        data_input[0,:] = data_scaled[0:20,0]   
        data_label[0,0] = data_scaled[22,0]
    
    for 2857 in (2858):
        data_input[2857,:] = data_scaled[2857:2879,0]   
        data_label[2857,0] = data_scaled[2879,0]
    '''
    # data_label = data_scaled[step_size+1:,0]
    # 划分数据集
    X_train = data_input[:-test_number]# (2658,20)
    Y_train = data_label[:-test_number]# (2685,1)
    X_test = data_input[-test_number:]# (200,20)
    Y_test = data_label[-test_number:]# (200,1)

    return  X_train, X_test, Y_train, Y_test, scaled_tool

def single_model(data,test_number,flag, input_step, pre_step):
    '''
    test_number, imfs_number, input_step, pre_step= 200, 15, 20, 2
    Parameters
    ----------
    data        使用IMFS函数重构成干净数据  (2880,0)
    flag        使用的模型
    Returns
    -------

    '''
    X_train, X_test, Y_train, Y_test, scaled_tool = Data_partitioning(data, test_number, input_step, pre_step) # 切片
    model = modelss(X_train, X_test, Y_train, Y_test, scaled_tool)
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
    data_pre = pre[:, 0]

    return data_pre


if __name__ == '__main__':
    test_number, imfs_number, input_step, pre_step= 200, 15, 20, 2
    data = pd.read_csv('10 min wind speed data.csv', header= None) # (2880,3)
    print(data.shape,data.iloc[:,2].values.shape)
    ssa_denoise = IMF_decomposition(data.iloc[:,2].values, imfs_number)  # 选取第三列数据  经过降噪之后
    np.savetxt('ssa_denoise_3.csv',ssa_denoise[-test_number:],delimiter = ',') # 使用IMFS函数重构成干净数据 ssa_denoise [2880]
    # pre_emd = single_model(emd_denoise, test_number, 'tcn_gru', input_step, pre_step)
    # pre_eemd = single_model(eemd_denoise, test_number, 'tcn_gru', input_step, pre_step)
    # pre_vmd = single_model(vmd_denoise, test_number, 'tcn_gru', input_step, pre_step)

    pre_ssa_tcn_gru = single_model(ssa_denoise, test_number, 'tcn_gru', input_step, pre_step)
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

    np.savetxt('pre_ssa_tcn_gru.csv', pre_ssa_tcn_gru, delimiter=',')
    # np.savetxt('pre_ssa_tcn_lstm.csv', pre_ssa_tcn_lstm, delimiter=',')
    # np.savetxt('pre_ssa_tcn_rnn.csv', pre_ssa_tcn_rnn, delimiter=',')
    # np.savetxt('pre_ssa_tcn_bpnn.csv', pre_ssa_tcn_bpnn, delimiter=',')
    # np.savetxt('pre_ssa_gru.csv', pre_ssa_gru, delimiter=',')
    # np.savetxt('pre_ssa_lstm.csv', pre_ssa_lstm, delimiter=',')
    # np.savetxt('pre_ssa_rnn.csv', pre_ssa_rnn, delimiter=',')
    # np.savetxt('pre_ssa_bpnn.csv', pre_ssa_bpnn, delimiter=',')
    Actual = ssa_denoise[-test_number:]    # ssa 后的 后两百条数据

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
