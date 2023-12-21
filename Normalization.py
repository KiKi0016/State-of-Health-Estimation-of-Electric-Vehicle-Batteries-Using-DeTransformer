import  matplotlib.pyplot as plt
import numpy as np 
import pickle


train_batch = pickle.load(open('Data/train_batch.pkl', 'rb'))


plt.figure(dpi=250)
plt.figure(1)
ax1 = plt.subplot(121)
ax1.plot(train_batch['b1c4']['summary']['cycle'], train_batch['b1c4']['summary']['QD'],color="r",linestyle = "--")
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity(Ah)')
plt.title('Capacity Degradation')
ax2 = plt.subplot(122)
ax2.plot(train_batch['b1c4']['summary']['cycle'], train_batch['b1c4']['summary']['QD']/1.1,color="y",linestyle = "-")
plt.xlabel('Cycle Number')
plt.ylabel('SOH(%)')
plt.title('SOH Degradation')
plt.show()
