import numpy as np
import matplotlib.pyplot as plt
import pickle


batch1 = pickle.load(open(r'./Data/batch1.pkl', 'rb'))
#remove batteries that do not reach 80% capacity
del batch1['b1c8']
del batch1['b1c10']
del batch1['b1c12']
del batch1['b1c13']
del batch1['b1c22']

numBat1 = len(batch1.keys())
print(numBat1)

batch2 = pickle.load(open(r'./Data/batch2.pkl','rb'))

# There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
# and put it with the correct cell from batch1
batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
add_len = [662, 981, 1060, 208, 482];

for i, bk in enumerate(batch1_keys):
    batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
    for j in batch1[bk]['summary'].keys():
        if j == 'cycle':
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
        else:
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
    last_cycle = len(batch1[bk]['cycles'].keys())
    for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
        batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]


del batch2['b2c7']
del batch2['b2c8']
del batch2['b2c9']
del batch2['b2c15']
del batch2['b2c16']        

numBat2 = len(batch2.keys())
print(numBat2)


batch3 = pickle.load(open(r'./Data/batch3.pkl','rb'))
# remove noisy channels from batch3
del batch3['b3c37']
del batch3['b3c2']
del batch3['b3c23']
del batch3['b3c32']
del batch3['b3c42']
del batch3['b3c43']


numBat3 = len(batch3.keys())
print(numBat3)


numBat = numBat1 + numBat2 + numBat3
print(numBat)


bat_dict = {**batch1, **batch2, **batch3}
#bat_dict = {**batch1, **batch2}
print(len(bat_dict))

plt.figure(dpi=250)
for i in bat_dict.keys():
    plt.plot(bat_dict[i]['summary']['cycle'], bat_dict[i]['summary']['QD'])  
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')
plt.title('Capacity Degradation')
#plt.savefig('./Data/total degradation.png')
plt.show()

'''
with open('train_batch.pkl','wb') as fp:
        pickle.dump(bat_dict,fp)
'''
'''
plt.figure(dpi=250)
plt.figure(1)
ax1 = plt.subplot(121)
for i in bat_dict.keys():
    ax1.plot(bat_dict[i]['summary']['cycle'], bat_dict[i]['summary']['QD'])
    plt.xlabel('Cycle Number')
    plt.ylabel('Discharge Capacity(Ah)')
    plt.title('Capacity Degradation')
ax2 = plt.subplot(122)
for i in bat_dict.keys():
    ax2.plot(bat_dict[i]['summary']['cycle'], bat_dict[i]['summary']['QD']/1.1)
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH(%)')
    plt.title('SOH Degradation')
plt.show()
'''


