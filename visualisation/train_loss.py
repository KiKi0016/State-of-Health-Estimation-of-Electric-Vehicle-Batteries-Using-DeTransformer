import csv
import matplotlib.pyplot as plt
import pandas as pd

Battery_list = ['b1c3', 'b1c4', 'b1c5', 'b1c6', 
                'b1c11', 'b1c21', 'b2c0', 'b2c2', 
                'b2c13', 'b2c14', 'b2c23', 'b2c47']

for i in Battery_list:
    data = pd.read_csv('Data/Train_Loss/Transformer_15/Battery'+ str(i) +'_Train Loss.csv')

    xdata = []
    y0, y1, y2, y3 = [], [], [], []
    xdata = data.loc[:,'Step']  
    y0 = data.loc[:,'Value0']   
    y1 = data.loc[:,'Value1']
    y2 = data.loc[:,'Value2']
    y3 = data.loc[:,'Value3']


    plt.figure(dpi=200)
    plt.plot(xdata, y0, label='Loss')
    plt.plot(xdata, y1, label='RPE')
    plt.plot(xdata, y2, label='RMSE')
    plt.plot(xdata, y3, label='MAE')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.ylim(0.0,2)
    plt.title('Train Loss')
    plt.legend(loc=0)
    plt.savefig('./Data/Train_Loss/Transformer_15/Train Loss_'+ str(i) +'.png')
    plt.show()

