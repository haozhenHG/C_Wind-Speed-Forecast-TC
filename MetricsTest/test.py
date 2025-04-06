#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/6 22:04
# @File ：test.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/4 16:09
# @File ：mteric.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
import numpy as np
import pandas as pd
# 1. 生成两个文件


# 生成从 1 到 100 的数组，步长为 1
test10 = np.arange(100, 1100, 10)  # 生成 1 到 100 的数组
print(test10)
np.savetxt("test10.csv", test10, delimiter=',')

test11= np.arange(105, 1100, 10)  # 生成 1 到 100 的数组
print(test11)
np.savetxt("test11.csv", test11, delimiter=',')

# 假设 y_true 是实际值，y_pred 是预测值
y_true = pd.read_csv('test10.csv').values
print(y_true.shape)
y_pred = pd.read_csv('test11.csv').values
print(y_pred.shape)


# 计算 MAE
mae = np.mean(np.abs(y_true - y_pred))
# 计算 MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# 计算 RMSE
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
'''
10   20   30   40   50   60   70   80   90  100  110  120  130  140   ....  
11   21   31   41   51   61   71   81   91  101  111  121  131  141   ....

MAE: 1.0
MAPE: 0.42296742602420406
RMSE: 1.0


10   20   30   40   50   60   70   80   90  100  110  120  130  140   ....  
12   22   32   42   52   62   72   82   92  102  112  122  132  142   ....

MAE: 2.0
MAPE: 0.8459348520484081
RMSE: 2.0
'''