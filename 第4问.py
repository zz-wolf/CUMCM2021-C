import pandas as pd
import numpy as np
from scipy.optimize import linprog
from copy import deepcopy
from time import perf_counter
import matplotlib.pyplot as plt


# 求解转运方案时使用的遗传算法(浮点数编码）
class GA:
    def __init__(self, nums, supply_want, omega, t_trans, cross_rate=0.8, mutation=0.003, iter_time=100, verbose=True):
        nums = np.array(nums)
        self.POP = nums  # popsize * 50 * 8
        self.supply_want = supply_want.reshape(-1, 1)  # 规划求得的希望供货数量
        self.omega = omega.reshape(-1, 1)  # 供应商本周的损失率
        self.t_trans = t_trans.reshape(-1, 1)  # 各供应商原材料向产品转化的转化率
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.POP_SIZE = len(nums)
        self.iter_time = iter_time
        self.verbose = verbose
        self.target_value = None

    def func(self):
        a = self.POP * self.supply_want  # a: 3darray (pop_size*50*8) 转运方案
        b = (a @ omega).reshape(self.POP.shape[0], 50)  # b: 2darray (pop_size*50) 各供应商损失数量
        c = b @ self.t_trans  # 2darray (pop_size*1) 各方案总损失值
        self.target_value = c

    def constrain(self):
        constrain1 = (np.around(self.POP.sum(axis=2), decimals=9) == 1.0).sum(axis=1) == 50
        constrain2 = ((self.POP * self.supply_want).sum(axis=1) <= 6000).sum(axis=1) == 8
        matrix3 = (self.POP[:, (self.supply_want <= 6000).reshape(-1)] == 1) | (
                self.POP[:, (self.supply_want <= 6000).reshape(-1)] == 0)
        constrain3 = matrix3.sum(axis=1).sum(axis=1) == (matrix3.shape[1] * matrix3.shape[2])
        matrix4 = (self.POP[:, (self.supply_want > 6000).reshape(-1)] < 1)
        constrain4 = matrix4.sum(axis=1).sum(axis=1) == (matrix4.shape[1] * matrix4.shape[2])
        return constrain1 & constrain2 & constrain3 & constrain4

    # 得到适应度并进行自然选择（轮盘赌）
    def get_fitness_and_select(self):
        # 计算适应度
        fitness = deepcopy(self.target_value)
        fitness = 1 / fitness
        prob = fitness / np.sum(fitness)  # 选择概率
        prob = prob.reshape(-1)

        # 自然选择（父代与子代满足约束条件的共同进行轮盘赌）
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP_SIZE, replace=True, p=prob)]

    # 染色体交叉
    def crossover_and_mutate(self):
        POP1 = deepcopy(self.POP)
        # 交叉
        for plan in POP1:
            if np.random.rand() < self.cross_rate:
                i = np.random.randint(0, POP1.shape[0], size=1)  # 选择要交叉的个体
                j = np.random.randint(0, POP1.shape[1], size=1)  # 选择要交叉的供应商方案
                plan[j], POP1[i, j] = deepcopy(POP1[i, j]), deepcopy(plan[j])

        # 变异1
        for i in range(POP1.shape[0]):
            for j in range(POP1.shape[1]):
                if np.random.rand() < self.mutation:
                    np.random.shuffle(POP1[i, j])

        # 变异2
        for i in range(POP1.shape[0]):
            for j in range(POP1.shape[1]):
                if (np.random.rand() < self.mutation) & (self.supply_want[j] >= 4000):
                    # 为供应量大于6000的供应商重新随机生成1组方案
                    n = np.random.randint((self.supply_want[j] // 6000) + 1, 9, size=1)
                    m = np.random.rand(int(n))
                    m = m / m.sum()
                    m = np.hstack((np.zeros(8 - n), m))
                    np.random.shuffle(m)
                    POP1[i, j] = deepcopy(m)

        self.POP = np.vstack((self.POP, POP1))

    # 1轮进化
    def evolution(self):
        self.crossover_and_mutate()
        self.POP = self.POP[self.constrain()]
        self.func()
        self.get_fitness_and_select()
        self.func()

    def fit(self):
        min_value = []  # 收集历次迭代最优目标值
        min_value_solution = []  # 收集历次迭代最优方案
        for i in range(self.iter_time):
            start = perf_counter()
            self.evolution()
            min_value.append(np.min(self.target_value))
            min_value_solution.append(self.POP[np.argmin(self.target_value)])
            end = perf_counter()
            if self.verbose:
                print('当前迭代{}/{}回,耗时{:.2f}s，本轮损耗量最小值{:.4f}'.format(i + 1, self.iter_time, end - start, min_value[-1]))

        # 展示历次迭代最大值,输出历代中最优结果
        min_value = np.array(min_value)
        plt.figure()
        plt.scatter(np.arange(1, len(min_value) + 1, 1), min_value)
        plt.show()
        best_solution = min_value_solution[np.argmin(min_value)]
        print('本周运货方案求解成功，损耗量最小值为{:.4f}'.format(np.min(min_value)))
        return best_solution


def t(str):
    if str == 'A':
        return 1 / 0.6
    elif str == 'B':
        return 1 / 0.66
    elif str == 'C':
        return 1 / 0.72


def p(str):
    if str == 'A':
        return 1.2 + 0.1
    elif str == 'B':
        return 1.1 + 0.1
    elif str == 'C':
        return 1.0 + 0.1


# 随机出一种可行的转运方案
def one_solution(supply_wanted):
    arr = np.zeros((50, 8))
    for index in np.argsort(supply_wanted)[::-1]:
        if supply_wanted[index] > 4000:
            n = np.random.randint((supply_wanted[index] // 6000) + 1, 9, size=1)
            m = np.random.rand(int(n))
            m = m / m.sum()
            m = np.hstack((np.zeros(8 - n), m))
            np.random.shuffle(m)
            arr[index] = m
        else:
            m = np.zeros(8)
            remain = 6000 - (arr * supply_wanted.reshape(-1, 1)).sum(axis=0)
            remain[remain < 0] = 0
            prob = remain / remain.sum()
            k = np.random.choice(np.arange(8), size=1, p=prob)
            m[k] += 1
            arr[index] = m
    arr = np.array(arr)
    return arr


def one_feasible_solu(supply_wanted):
    arr = one_solution(supply_wanted)
    while ((arr * supply_wanted.reshape(-1, 1)).sum(axis=0) <= 6000).sum() != 8:
        arr = one_solution(supply_wanted)
    return arr


# 读取数据
data_df = pd.read_excel('供应商供货量.xlsx', sheet_name=0)
data = np.array(data_df)
data2_df = pd.read_excel('转运商数据.xlsx')
data2 = np.array(data2_df)[:, 1:]

t_trans = np.array([t(str) for str in data[:, 1]])  # 各供应商的原材料转化率
price = np.array([p(str) for str in data[:, 1]])  # 各供货商单位原材料总价格（材料+转运+储存）
omega = [data2[i][data2[i] != 0].mean() for i in range(8)]  # 计算各转运商的平均损耗率
omega = np.array(omega) / 100  # 将平均损耗率百分数转化为小数

# 求各供应商的订货/供货比（便于后文通过供货量推算订货量）
data3_df = pd.read_excel('供应商相关数据.xlsx', sheet_name=0)
data3 = np.array(data3_df)[:, 3:]
data3[data3 > 2] = 2  # 异常值处理，当x>2时视为异常值，以最大值2代替
# 分别计算每年（48周）平均值
year1_mean = np.array([data3[i, 0:48][data3[i, 0:48] != 0].mean() for i in range(data3.shape[0])])
year2_mean = np.array([data3[i, 48:96][data3[i, 48:96] != 0].mean() for i in range(data3.shape[0])])
year3_mean = np.array([data3[i, 96:144][data3[i, 96:144] != 0].mean() for i in range(data3.shape[0])])
year4_mean = np.array([data3[i, 144:192][data3[i, 144:192] != 0].mean() for i in range(data3.shape[0])])
year5_mean = np.array([data3[i, 192:240][data3[i, 192:240] != 0].mean() for i in range(data3.shape[0])])
order_supply = (1 * year1_mean + 2 * year2_mean + 3 * year3_mean + 4 * year4_mean + 5 * year5_mean) / (
        1 + 2 + 3 + 4 + 5)

order_all_week = np.zeros(50).reshape(-1, 1)  # 收集每周订货方案
trans_all_week = np.zeros(50).reshape(-1, 1)  # 收集每周转运方案
inventory = 0  # 第0周时库存为0
for week in range(24):
    # 求解订货方案
    need = 67118 - inventory  # 计算本周需求（本周库存需求-上周库存）
    max_bound = data[:, week + 4]  # 求得本周各供应商的最大供货量
    bounds = [(0, x) for x in max_bound]
    res = linprog(c=price, A_ub=-t_trans.reshape(1, -1), b_ub=-np.array(need), bounds=bounds)

    supply_wanted = res.x
    order_want = supply_wanted * order_supply
    order_all_week = np.hstack((order_all_week, order_want.reshape(-1, 1)))

    # 求解运输方案
    nums = []
    start = perf_counter()
    for i in range(2000):
        arr = one_feasible_solu(supply_wanted)
        nums.append(arr)
    end = perf_counter()
    print('遗传算法初始值计算完毕，用时{:.2f}s'.format(end - start))
    nums = np.array(nums)
    ga = GA(nums=nums, supply_want=supply_wanted, omega=omega, t_trans=t_trans, cross_rate=0.8, mutation=0.003,
            iter_time=150, verbose=True)
    best_solution = ga.fit()
    print('第{}周转运方案以遗传算法求解成功'.format(week + 1))
    best_solution = best_solution * supply_wanted.reshape(-1, 1)
    trans_all_week = np.hstack((trans_all_week, best_solution))

    # 计算本周库存
    inventory = inventory + (best_solution @ (1 - omega).reshape(-1, 1) * t_trans.reshape(-1, 1)).sum() - 33559

# 输出订货情况
pd.DataFrame(np.hstack((data[:, 0].reshape(-1, 1), order_all_week[:, 1:]))).to_excel('未来24周订货情况.xlsx')
# 输出转运情况
pd.DataFrame(np.hstack((data[:, 0].reshape(-1, 1), trans_all_week[:, 1:]))).to_excel('未来24周转运情况.xlsx')
