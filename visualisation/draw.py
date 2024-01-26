import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
'''
name_list = ['Training set', 'Test set']
'''
name_list = ['Start Point = 81', 'Start Point = 161', 'Start Point = 321']

'''
num_list0 = [0.1228, 0.3169]
num_list1 = [0.0647, 0.0779]
num_list2 = [0.0516, 0.0577]
'''
'''
num_list0 = [0.1228, 0.1587, 0.1649]
num_list1 = [0.0647, 0.0608, 0.0514]
num_list2 = [0.0516, 0.0473, 0.0412]
'''
num_list0 = [0.3169, 0.3436, 0.8619]
num_list1 = [0.0779, 0.0803, 0.0404]
num_list2 = [0.0577, 0.0607, 0.0239]


x =list(range(len(num_list0)))
total_width, n = 0.9, 3
width = total_width / n


plt.figure(dpi=200)
plt.bar(x, num_list0, width=0.9*width, label='RPE', fc = '#00BFFF')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=0.9*width, label='RMSE', tick_label = name_list, fc = '#D2691E')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=0.9*width, label='MAE', fc = '#32CD32')
plt.ylim([0, 1])
for i in range(len(x)):
    plt.text(x[i] - 2*width, num_list0[i] + 0.01, num_list0[i], ha='center')
    plt.text(x[i] - width, num_list1[i] + 0.01, num_list1[i], ha='center')
    plt.text(x[i] , num_list2[i] + 0.01, num_list2[i], ha='center')
plt.legend(loc=0)
#plt.savefig('./Data/Evaluation/Performance on the trainset.png')
#plt.savefig('./Data/Evaluation/Performance on the training and testset.png')
#plt.savefig('./Data/Evaluation/Different start point on the training set.png')
plt.savefig('./Data/Evaluation/Different start point on the test set.png')
plt.show()