import numpy as np
import pandas as pd
from copy import deepcopy

# 数据读取
data1 = np.array(pd.read_excel('附件1 近5年402家供应商的相关数据.xlsx', sheet_name=0))
data2 = np.array(pd.read_excel('附件1 近5年402家供应商的相关数据.xlsx', sheet_name=1))
data_order = data1[:, 2:]
data_supply = data2[:, 2:]
data_supplier = data1[:, 0:2]

# 计算各供应商订货量/供货量能够生产的产品数
data_order2 = np.zeros(data_order.shape)
for i in range(data_order.shape[0]):
    if data_supplier[i, 1] == 'A':
        data_order2[i] = data_order[i] * (1 / 0.6)
    elif data_supplier[i, 1] == 'B':
        data_order2[i] = data_order[i] * (1 / 0.66)
    else:
        data_order2[i] = data_order[i] * (1 / 0.72)
data_order2.sum() / 240
data_supply2 = np.zeros(data_supply.shape)
for i in range(data_supply.shape[0]):
    if data_supplier[i, 1] == 'A':
        data_supply2[i] = data_supply[i] * (1 / 0.6)
    elif data_supplier[i, 1] == 'B':
        data_supply2[i] = data_supply[i] * (1 / 0.66)
    else:
        data_supply2[i] = data_supply[i] * (1 / 0.72)

# 供货数量
supply_num = data_supply2.sum(axis=1)
# 供货稳定指数 1/(供货量变异系数+1）
supply_stability = np.array(
    [np.std(data_supply2[k][data_order2[k] != 0]) / np.mean(data_supply2[k][data_order2[k] != 0])
     for k in range(402)])
supply_stability2 = 1 / (supply_stability + 1)
# 供货偏移指数 1/(供货量与订货量差的平方平均数+1)
diff = data_supply2 - data_order2
supply_shift = []
for k in range(402):
    shift_i = np.mean(np.square(diff[k][data_order2[k] != 0] / data_order2[k][data_order2[k] != 0]))
    supply_shift.append(shift_i)
supply_shift = np.array(supply_shift)
supply_shift2 = 1 / (supply_shift + 1)
# 订货数量
order_num = data_order2.sum(axis=1)
# 订货稳定指数  1/(订货量的变异系数+1)
order_stability = np.array(
    [np.std(data_order2[k][data_order2[k] != 0]) / np.mean(data_order2[k][data_order2[k] != 0])
     for k in range(402)])
order_stability2 = 1 / (order_stability + 1)
# 供应商占用率 1-闲置率
vacancy_rate = (data_order == 0).sum(axis=1) / 240
vacancy_rate2 = 1 - vacancy_rate
# 供应商守约率 1-违约
default_rate = ((data_order > 0) & (data_supply == 0)).sum(axis=1) / (data_order != 0).sum(axis=1)
default_rate2 = 1 - default_rate
# 重要订单接收频次
important_freq = np.zeros(402)
for i in range(data_order2.shape[1]):
    index = np.argsort(data_order2[:, i])[::-1][0:20]
    important_freq[index] += 1
# 供应商细分市场份额
all = np.zeros(402)
for i in range(len(all)):
    if data_supplier[i, 1] == 'A':
        all[i] = np.sum(data_supply2[data_supplier[:, 1] == "A", :])
    elif data_supplier[i, 1] == 'B':
        all[i] = np.sum(data_supply2[data_supplier[:, 1] == "B", :])
    else:
        all[i] = np.sum(data_supply2[data_supplier[:, 1] == "C", :])
segmentation_market_share = data_supply2.sum(axis=1) / all

# 整合各特征
data_feature = np.vstack([supply_num, supply_stability2, supply_shift2, order_num, order_stability2, important_freq,
                          segmentation_market_share, vacancy_rate2, default_rate2]).T

# 特征数据输出
data_out = pd.DataFrame(data_feature)
data_out.columns = ['供货数量', '供货稳定指数', '供货偏移指数', '订货数量', '订货稳定指数', '重要订单接受频次',
                    '供应商细分市场份额', '供应商占用率', '供应商守约率']
data_out.to_excel('第1问参数数值.xlsx')

# 特征归一化处理
from sklearn.preprocessing import MinMaxScaler

feature_scaled = deepcopy(data_feature)
feature_scaled[:, [0, 3, 5]] = MinMaxScaler().fit_transform(feature_scaled[:, [0, 3, 5]])

# 主成分加权
from sklearn.decomposition import PCA

pca = PCA(svd_solver='full')
feature_pca = pca.fit_transform(feature_scaled)
variance = pca.explained_variance_ratio_.reshape(-1, 1)
supplier_score = feature_pca @ variance
components = pca.components_
pd.DataFrame(supplier_score).to_excel('供应商得分.xlsx')  # 输出402家企业加权得分
pd.DataFrame(np.hstack((variance, components))).to_excel('主成分.xlsx')  # 输出主成分的参数

# 计算各企业最大供应量
max_offer = data_supply2.max(axis=1).reshape(-1, 1)
a = np.hstack((supplier_score, max_offer))
a = a[np.argsort(a[:, 0])[::-1]]  # a从大到小排序
m = np.array([a[:i, 1].sum() for i in range(50)])
