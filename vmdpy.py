# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:24:58 2019

@author: Vinícius Rezende Carvalho
"""
import numpy as np
import pandas as pd


def  VMD(f, alpha, tau, K, DC, init, tol):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
    
    
    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """
    
    if len(f)%2:
       f = f[:-1]

    # Period and sampling frequency of input signal
    fs = 1./len(f)
    
    ltemp = len(f)//2 
    fMirr =  np.append(np.flip(f[:ltemp],axis = 0),f)  
    fMirr = np.append(fMirr,np.flip(f[-ltemp:],axis = 0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1,T+1)/T  
    
    # Spectral Domain discretization
    freqs = t-0.5-(1/T)

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha*np.ones(K)
    
    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat) #copy f_hat
    f_hat_plus[:T//2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])


    if init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K)*(i)
    elif init == 2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
    else:
        omega_plus[0,:] = 0
            
    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0,0] = 0
    
    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)
    
    # other inits
    uDiff = tol+np.spacing(1) # update step
    n = 0 # loop counter
    sum_uk = 0 # accumulator
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)    

    #*** Main loop for iterative updates***

    while ( uDiff > tol and  n < Niter-1 ): # not converged and below iterations limit
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,0]
        
        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1.+Alpha[k]*(freqs - omega_plus[n,k])**2)
        
        # update first omega if not held at 0
        if not(DC):
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)

        # update of any other mode
        for k in np.arange(1,K):
            #accumulator
            sum_uk = u_hat_plus[n+1,:,k-1] + sum_uk - u_hat_plus[n,:,k]
            # mode spectrum
            u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n,k])**2)
            # center frequencies
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)
            
        # Dual ascent
        lambda_hat[n+1,:] = lambda_hat[n,:] + tau*(np.sum(u_hat_plus[n+1,:,:],axis = 1) - f_hat_plus)
        
        # loop counter
        n = n+1
        
        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1/T)*np.dot((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i]),np.conj((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i])))

        uDiff = np.abs(uDiff)        
            
    #Postprocessing and cleanup
    
    #discard empty space if converged early
    Niter = np.min([Niter,n])
    omega = omega_plus[:Niter,:]
    
    idxs = np.flip(np.arange(1,T//2+1),axis = 0)
    # Signal reconstruction
    u_hat = np.zeros([T, K],dtype = complex)
    u_hat[T//2:T,:] = u_hat_plus[Niter-1,T//2:T,:]
    u_hat[idxs,:] = np.conj(u_hat_plus[Niter-1,T//2:T,:])
    u_hat[0,:] = np.conj(u_hat[-1,:])    
    
    u = np.zeros([K,len(t)])
    for k in range(K):
        u[k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))
        
    # remove mirror part
    u = u[:,T//4:3*T//4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1],K],dtype = complex)
    for k in range(K):
        u_hat[:,k]=np.fft.fftshift(np.fft.fft(u[k,:]))

    return u, u_hat, omega



def determine_K_based_on_frequency(f, alpha, tau, DC, init, tol, max_K=20, frequency_threshold=0.01):
    """
    根据中心频率确定最佳的 K 值。

    参数:
    f: 待分解的时域信号
    alpha: 数据保真约束的平衡参数
    tau: 对偶上升的时间步长
    DC: 布尔值，若为 True，则第一个模态将被设置并保持在直流（0 频率）
    init: 初始化模态中心频率的方式
    tol: 收敛准则的容差
    max_K: 最大尝试的 K 值
    frequency_threshold: 中心频率差异的阈值

    返回:
    最佳的 K 值
    """
    best_K = 2
    previous_omega = None

    for K in range(2, max_K + 1):
        # 进行 VMD 分解
        u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
        current_omega = omega[-1, :]  # 取最后一次迭代的中心频率

        if previous_omega is not None:
            # 计算新增加的模态的中心频率与已有模态中心频率的最小差异
            new_frequency_differences = []
            for new_freq in current_omega[len(previous_omega):]:
                min_diff = np.min(np.abs(new_freq - previous_omega))
                new_frequency_differences.append(min_diff)

            # 如果新增加的模态的中心频率与已有模态中心频率的最小差异都小于阈值，则停止增加 K
            if all(np.array(new_frequency_differences) < frequency_threshold):
                break

        best_K = K
        previous_omega = current_omega

    return best_K

if __name__ == '__main__':
    # 设置 VMD 参数
    # alpha = 2000
    # tau = 0
    # DC = False
    # init = 1
    # tol = 1e-6

    data = pd.read_csv('./jupyter/American_final_to_model.csv',index_col=0)

    print(data['load'].values.shape)

    alpha = 5000  # 带宽约束参数
    tau = 0.0  # 噪声容忍度（0表示严格保真）

    DC = 0  # 不保留直流分量
    init = 1  # 初始化模式
    tol = 1e-8  # 停止迭代阈值
    # 确定最佳的 K 值   中心频率法
    # best_K = determine_K_based_on_frequency(data['load'].values, alpha, tau, DC, init, tol)
    # print(f"最佳的 K 值为: {best_K}")   # 最佳的 K 值为: 6


    # ----------信息准则法
    # 可以使用信息准则（如 AIC、BIC）来评估不同 K 值下分解结果的优劣。信息准则越小，说明模型越好。
    # 尝试不同的 K 值
    signal = data['load'].values
    # K_values = range(2, 20)
    # AIC_values = []
    # for K in K_values:
    #     u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    #
    #     # 调整 u 的长度
    #     if len(signal) != u.shape[1]:
    #         if len(signal) > u.shape[1]:
    #             # 如果信号长度大于 u 的长度，在 u 后面补零
    #             padding = len(signal) - u.shape[1]
    #             u = np.pad(u, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    #         else:
    #             # 如果信号长度小于 u 的长度，截断 u
    #             u = u[:, :len(signal)]
    #
    #     residuals = data['load'].values - np.sum(u, axis=0)
    #     n = len(data.values)
    #     p = K  # 参数数量
    #     likelihood = -0.5 * n * np.log(np.var(residuals))
    #     AIC = 2 * p - 2 * likelihood
    #     AIC_values.append(AIC)
    #
    # best_K = K_values[np.argmin(AIC_values)]
    # print(f'Best K value: {best_K}') #-------Best K value: 19


    # ---基于能量分布的方法
    # 尝试不同的 K 值  分析分解后各模态的能量分布，当新增模态的能量占比低于某个阈值时，认为当前的 K 值是合适的。
    K_max = 20
    energy_threshold = 0.01

    for K in range(1, K_max + 1):
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        energies = np.sum(u ** 2, axis=1)
        total_energy = np.sum(energies)
        energy_ratios = energies / total_energy
        if K > 1 and energy_ratios[-1] < energy_threshold:
            best_K = K - 1
            break
    else:
        best_K = K_max

    print(f'Best K value : {best_K}')