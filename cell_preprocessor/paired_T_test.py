import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns           #使用seaborn 库画直方图验证结果
import math
from scipy.stats import ttest_rel


labels = [[95.55, 3.38, 0.01], [97.58, 2.32, 0.01], [92.07, 7.68, 0.21], [85.18, 9.77, 0.01]]
Neu_AI = [92.02, 93.95, 89.44, 82.78]
Neu_FCM = [95.55, 97.58, 92.07, 85.18]

Mac_AI = [4.47, 3.21, 8.54, 10.99]
Mac_FCM = [3.38, 2.32, 7.68, 9.77]

Lym_AI = [2.48, 2.32, 2.77, 2.31]
Lym_FCM = [0.01, 0.01, 0.21, 0.01]

# 求均值与标准差
mean1, sd1 = 0, 0
for i in Neu_AI:
    mean1 += i
mean1 /= 4
for i in Neu_AI:
    sd1 += math.pow(i - mean1, 2)
sd1 /= 4
sd1 = math.sqrt(sd1)
# # 求均值与标准差
mean2, sd2 = 0, 0
for i in Neu_FCM:
    mean2 += i
mean2 /= 4
for i in Neu_FCM:
    sd2 += math.pow(i - mean2, 2)
sd2 /= 4
sd2 = math.sqrt(sd2)
print('中性粒结果')
print(mean1, sd1)
print(mean2, sd2)
print(mean1 - mean2, sd1 - sd2)
print(ttest_rel(Neu_AI, Neu_FCM))

# 求均值与标准差
mean1, sd1 = 0, 0
for i in Mac_AI:
    mean1 += i
mean1 /= 4
for i in Mac_AI:
    sd1 += math.pow(i - mean1, 2)
sd1 /= 4
sd1 = math.sqrt(sd1)
# # 求均值与标准差
mean2, sd2 = 0, 0
for i in Mac_FCM:
    mean2 += i
mean2 /= 4
for i in Mac_FCM:
    sd2 += math.pow(i - mean2, 2)
sd2 /= 4
sd2 = math.sqrt(sd2)
print('单核巨噬结果')
print(mean1, sd1)
print(mean2, sd2)
print(mean1 - mean2, sd1 - sd2)
print(ttest_rel(Mac_AI, Mac_FCM))


# 求均值与标准差
mean1, sd1 = 0, 0
for i in Lym_AI:
    mean1 += i
mean1 /= 4
for i in Lym_AI:
    sd1 += math.pow(i - mean1, 2)
sd1 /= 4
sd1 = math.sqrt(sd1)
# # 求均值与标准差
mean2, sd2 = 0, 0
for i in Lym_FCM:
    mean2 += i
mean2 /= 4
for i in Lym_FCM:
    sd2 += math.pow(i - mean2, 2)
sd2 /= 4
sd2 = math.sqrt(sd2)
print('淋巴结果')
print(mean1, sd1)
print(mean2, sd2)
print(mean1 - mean2, sd1 - sd2)
print(ttest_rel(Lym_AI, Lym_FCM))