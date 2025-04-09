1. VMD分解后长度不一致问题
    在VMD分解过程中出现输出长度与输入长度不一致的情况（如输入21769，输出21768），通常由以下原因及解决方案构成
    FFT对称性处理： VMD内部使用FFT进行频域分解，当信号长度为奇数时，某些库实现可能自动调整长度为偶数（如截断末尾1点），导致输出少1个点。
   ```python
    from scipy.interpolate import interp1d

    def VMD(self):
        data = self.data
        alpha, tau, length, DC, init, tol = 5000, 0, len(data), 0, 1, 1e-8
        u, u_hat, omega = VMD(data, alpha, tau, length, DC, init, tol)
    
        # 对齐长度（线性插值）
        if u.shape[1] != len(data):
            x_old = np.linspace(0, 1, u.shape[1])
            x_new = np.linspace(0, 1, len(data))
            f = interp1d(x_old, u, axis=1, kind='linear')
            u_aligned = f(x_new)
            return u_aligned.T
        else:
            return u.T
   ```
2. 