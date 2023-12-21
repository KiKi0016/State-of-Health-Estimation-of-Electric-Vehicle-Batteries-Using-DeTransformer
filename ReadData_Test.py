import numpy as np
import matplotlib.pyplot as plt
import pickle
'''
Battery_list = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4', 'b1c5', 'b1c6', 
                'b1c7', 'b1c9', 'b1c11', 'b1c14', 'b1c15', 'b1c16', 
                'b1c17', 'b1c18', 'b1c19', 'b1c20', 'b1c21', 'b1c23', 
                'b1c24', 'b1c25', 'b1c26', 'b1c27', 'b1c28', 'b1c29', 
                'b1c30', 'b1c31', 'b1c32', 'b1c33', 'b1c34', 'b1c35', 
                'b1c36', 'b1c37', 'b1c38', 'b1c39', 'b1c40', 'b1c41', 
                'b1c42', 'b1c43', 'b1c44', 'b1c45',
                'b2c0', 'b2c1', 
                'b2c2', 'b2c3', 'b2c4', 'b2c5', 'b2c6', 
                'b2c10', 'b2c11', 'b2c12', 'b2c13', 'b2c14', 'b2c17', 
                'b2c18', 'b2c19', 'b2c20', 'b2c21', 'b2c22', 'b2c23', 
                'b2c24', 'b2c25', 'b2c26', 'b2c27', 'b2c28', 'b2c29', 
                'b2c30', 'b2c31', 'b2c32', 'b2c33', 'b2c34', 'b2c35',
                'b2c36', 'b2c37', 'b2c38', 'b2c39', 'b2c40', 'b2c41', 
                'b2c42', 'b2c43', 'b2c44', 'b2c45', 'b2c46', 'b2c47']
 
'''           
Battery_list = ['b3c0', 'b3c1', 'b3c3', 'b3c4', 'b3c5', 'b3c6', 'b3c7',
                'b3c8', 'b3c9', 'b3c10', 'b3c11', 'b3c12', 'b3c13', 
                'b3c14', 'b3c15', 'b3c16', 'b3c17', 'b3c18', 'b3c19', 
                'b3c20', 'b3c21', 'b3c22',  'b3c24', 'b3c25', 'b3c26', 
                'b3c27', 'b3c28', 'b3c29', 'b3c30', 'b3c31', 'b3c33', 
                'b3c34', 'b3c35', 'b3c36', 'b3c38', 'b3c39', 'b3c40', 
                'b3c41', 'b3c44', 'b3c45']


train_batch = pickle.load(open('Data/train_batch.pkl', 'rb'))
'''
for i in Battery_list:
    #print('Battery_list: | length:'.format(i,len(train_batch[i]['summary']['QD'])))
    print('Battery_list: {} | length: {}'.format(i, len(train_batch[i]['summary']['QD'])))
'''
predict_list0 = []
predict_list1 = []
predict_list2 = []
'''
for num in range(84):
    predict_list.append(pickle.load(open('Data/Transformer_4/predict_list_main_4_'+ str(num) +'.pkl', 'rb')))
'''
for num in range(40):
    predict_list0.append(pickle.load(open('Data/Test_15/predict_list_test_15_'+ str(num) +'.pkl', 'rb')))
    predict_list1.append(pickle.load(open('Data/Test_16/predict_list_test_16_'+ str(num) +'.pkl', 'rb')))
    predict_list2.append(pickle.load(open('Data/Test_17/predict_list_test_17_'+ str(num) +'.pkl', 'rb')))


count = 0
start_index0 = 81
start_index1 = 161
start_index2 = 321
'''
for i in Battery_list:
    plt.figure(dpi=200)
    plt.plot(train_batch[i]['summary']['cycle'], train_batch[i]['summary']['QD']/1.1, ls = '-', label='Real') 
    x = np.arange(0, len(train_batch[i]['summary']['cycle']))
    plt.plot(x[start_index:],np.array(predict_list[count][-2])/1.1, ls = 'dotted', label='Prediction')
    plt.plot(train_batch[i]['summary']['cycle'], [0.8]*len(train_batch[i]['summary']['cycle']), c ='r', ls = '-.', label='Failure threshold')  
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH(%)')
    plt.title('SOH Prediction')
    plt.legend(loc=0,ncol=3)
    plt.savefig('./Data/Transformer_4/SOH_'+ str(i) +'.png')
    plt.show()
    count += 1
print(count)
'''

for i in Battery_list:
    plt.figure(dpi=200)
    plt.plot(train_batch[i]['summary']['cycle'], train_batch[i]['summary']['QD']/1.1, ls = '-', label='Real') 
    x = np.arange(0, len(train_batch[i]['summary']['cycle']))
    plt.plot(x[start_index0:],np.array(predict_list0[count])/1.1, ls = 'dotted', label='Start Point = 81')
    plt.plot(x[start_index1:],np.array(predict_list1[count])/1.1, ls = 'solid', label='Start Point = 161')
    plt.plot(x[start_index2:],np.array(predict_list2[count])/1.1, c ='c', ls = '--', label='Start Point = 321')
    plt.plot(train_batch[i]['summary']['cycle'], [0.8]*len(train_batch[i]['summary']['cycle']), c ='r', ls = '-.', label='Failure threshold')  
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH(%)')
    plt.title('SOH Prediction')
    plt.legend(loc=0)
    plt.savefig('./Data/Evaluation/SOH_'+ str(i) +'.png')
    plt.show()
    count += 1
print(count)
