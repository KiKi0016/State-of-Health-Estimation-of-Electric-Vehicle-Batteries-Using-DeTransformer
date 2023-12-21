import numpy as np
import matplotlib.pyplot as plt
import pickle

Battery_list = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4', 'b1c5', 'b1c6', 
                'b1c7', 'b1c9', 'b1c11', 'b1c14', 'b1c15', 'b1c16', 
                'b1c17', 'b1c18', 'b1c19', 'b1c20', 'b1c21', 'b1c23', 
                'b1c24', 'b1c25', 'b1c26', 'b1c27', 'b1c28', 'b1c29', 
                'b1c30', 'b1c31', 'b1c32', 'b1c33', 'b1c34', 'b1c35', 
                'b1c36', 'b1c37', 'b1c38', 'b1c39', 'b1c40', 'b1c41', 
                'b1c42', 'b1c43', 'b1c44', 'b1c45']


train_batch = pickle.load(open('Data/train_batch.pkl', 'rb'))


for i in Battery_list:
    fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()          
    ax1.plot(train_batch[i]['cycles']['1']['Qc']/1.1, train_batch[i]['cycles']['1']['V'], c ='b', ls = '-', label='Voltage')
    ax2.plot(train_batch[i]['cycles']['1']['Qc']/1.1, train_batch[i]['cycles']['1']['I'], c ='g', ls = ':', label='Current')
    ax1.set_xlabel('SOC(%)')    
    ax1.set_ylabel('Voltage(V)')   
    ax2.set_ylabel('Current(mA)')   
    fig.legend(loc=1, bbox_to_anchor=(0.64,0.2), bbox_transform=ax1.transAxes)
    fig.suptitle('Charge Policy_'+ str(i))
    plt.savefig('./Data/Charge_Policy/Charge Policy_'+ str(i) +'.png')
    plt.show()


