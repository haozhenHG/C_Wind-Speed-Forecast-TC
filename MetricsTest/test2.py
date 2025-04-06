import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 生成数据集
np.random.seed(42)
data1 = np.random.normal(50, 10, 100)
print(data1[:10])
'''
[54.96714153 48.61735699 56.47688538 65.23029856 47.65846625 47.65863043
 65.79212816 57.67434729 45.30525614 55.42560044]
'''
data2 = data1 + np.random.normal(0, 2, 100)
'''
[52.13640005 47.77606634 55.79145635 63.62574403 47.33589483 48.46673214
 69.56449996 58.02350292 45.82035692 55.2767086 ]
'''
print(data2[:10])


# 计算归一化前的误差指标
mae_before = mean_absolute_error(data1, data2)
mape_before = mean_absolute_percentage_error(data1, data2)
rmse_before = np.sqrt(mean_squared_error(data1, data2))

print(f"归一化前的MAE: {mae_before:.2f}")
print(f"归一化前的MAPE: {mape_before:.2f}%")
print(f"归一化前的RMSE: {rmse_before:.2f}")
'''
归一化前的MAE: 1.51
归一化前的MAPE: 0.03%
归一化前的RMSE: 1.90
'''
print('#'*30)
# 归一化数据集
scaler = MinMaxScaler()
data1_normalized = scaler.fit_transform(data1.reshape(-1, 1)).flatten()
print(data1_normalized[:10])
'''
[0.69687903 0.55488996 0.73063878 0.92637598 0.53344797 0.53345164
 0.93893919 0.75741552 0.48082726 0.70713074]
'''
data2_normalized = scaler.fit_transform(data2.reshape(-1, 1)).flatten()
print(data2_normalized[:10])
'''
[0.61449566 0.51804638 0.69534441 0.8686365  0.50830992 0.5333237
 1.         0.74471661 0.47478668 0.68395834]
'''
# 计算归一化后的误差指标
mae_after = mean_absolute_error(data1_normalized, data2_normalized)
mape_after = mean_absolute_percentage_error(data1_normalized, data2_normalized)
rmse_after = np.sqrt(mean_squared_error(data1_normalized, data2_normalized))

print(f"归一化后的MAE: {mae_after:.2f}")
print(f"归一化后的MAPE: {mape_after:.2f}%")
print(f"归一化后的RMSE: {rmse_after:.2f}")
'''
归一化后的MAE: 0.04
归一化后的MAPE: 0.08%
归一化后的RMSE: 0.05
'''