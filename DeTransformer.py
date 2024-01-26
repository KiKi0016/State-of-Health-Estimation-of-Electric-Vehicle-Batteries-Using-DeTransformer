import os
import random
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import scipy.io
from datetime import datetime
import math
from math import sqrt

import time
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

 
import numpy as np
import pandas as pd
import torch.nn.functional as F
import pickle

from tqdm import tqdm
from tensorboardX import SummaryWriter
import copy

# pylint: disable=arguments-differ

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# convert str to datatime 
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# load data
batch1 = pickle.load(open(r'Data/batch1.pkl', 'rb'))
#remove batteries that do not reach 80% capacity
del batch1['b1c8']
del batch1['b1c10']
del batch1['b1c12']
del batch1['b1c13']
del batch1['b1c22']

numBat1 = len(batch1.keys())

batch2 = pickle.load(open(r'Data/batch2.pkl','rb'))

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


#bat_dict = {**batch1}
bat_dict = {**batch1, **batch2}
#print(bat_dict.keys())
#print(len(bat_dict))

'''
for i in bat_dict.keys():
    plt.plot(bat_dict[i]['summary']['cycle'], bat_dict[i]['summary']['QD'])
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')
plt.show()
'''

# get capacity data
def getBatteryCapacity(bat_dict, name):
    cycle, capacity = [], []
    capacity = bat_dict[name]['summary']['QD']
    if len(capacity) < 400:
    # 计算需要填充的0的数量
        padding_length = 400 - len(capacity)
    # 使用 [0] * padding_length 创建一个包含0的列表
        padding = [0] * padding_length
    # 将原始列表和0填充列表连接起来
        #capacity = np.r_[capacity, padding]
    #print(len(capacity))
    
    for i in range(len(capacity)):
        if name in bat_dict.keys():
            cycle.append(i)
    return [cycle, capacity]


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
print(len(Battery_list))
Battery = {}
for name in Battery_list:
    Battery[name] = getBatteryCapacity(bat_dict, name)


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)
        #arr_x, arr_y = np.array(x), np.array(y)
    #return arr_x.astype(np.float32), arr_y.astype(np.float32)
    return np.array(x), np.array(y)

def split_dataset(data_sequence, train_ratio=0.0, capacity_threshold=0.0):
    if capacity_threshold > 0:
        max_capacity = max(data_sequence)
        capacity = max_capacity * capacity_threshold
        point = [i for i in range(len(data_sequence)) if data_sequence[i] < capacity]
    else:
        point = int(train_ratio + 1)
        if 0 < train_ratio <= 1:
            point = int(len(data_sequence) * train_ratio)
    train_data, test_data = data_sequence[:point], data_sequence[point:]
    return train_data, test_data


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, window_size=8):
    data_sequence=data_dict[name][1]
    #print(len(data_sequence))
    train_data, test_data = data_sequence[:5*window_size+1], data_sequence[5*window_size+1:]
    X_train, y_train = build_sequences(text=train_data, window_size=window_size)
    #print(X_train.dtype)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v[1], window_size=window_size)
            X_train, y_train = np.r_[X_train, data_x], np.r_[y_train, data_y]
    #print(X_train.dtype,y_train.dtype)
    print(len(test_data))
            
    return X_train, y_train, list(train_data), list(test_data)


def relative_positional_error(y_test, y_predict, threshold):
    true_rpe, pred_rpe = len(y_test), 0 #len(y_predict)
    for i in range(1,len(y_test)):
        if y_test[i-1] >= threshold and threshold >= y_test[i]:
            true_rpe = i 
            break
    for i in range(1,len(y_predict)):
        if y_predict[i-1] >= threshold and threshold >= y_predict[i]:
            pred_rpe = i 
            break
    return abs(true_rpe - pred_rpe)/true_rpe

def evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    #precision = precision_score(y_test, y_predict, average=None)
    return rmse


def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Autoencoder(nn.Module):
    def __init__(self, input_size=16, hidden_dim=8, noise_level=0.01):
        super(Autoencoder, self).__init__()
        self.input_size, self.hidden_dim, self.noise_level = input_size, hidden_dim, noise_level
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_size)
        
    def encoder(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1
    
    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x
    
    def decoder(self, x):
        h2 = self.fc2(x)
        return h2
    
    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode

 
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        
        super(PositionalEncoding, self).__init__()      
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x

          
 
class Net(nn.Module):
    def __init__(self, feature_size=16, hidden_dim=32, num_layers=1, nhead=8, dropout=0.0, noise_level=0.01):
        super(Net, self).__init__()
        self.auto_hidden = int(feature_size/2)
        input_size = self.auto_hidden 
        self.pos = PositionalEncoding(d_model=input_size, max_len=input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(input_size, 1)
        self.autoencoder = Autoencoder(input_size=feature_size, hidden_dim=self.auto_hidden, noise_level=noise_level)
 
    def forward(self, x): 
        batch_size, feature_num, feature_size  = x.shape 
        encode, decode = self.autoencoder(x.reshape(batch_size, -1))# batch_size*seq_len
        out = encode.reshape(batch_size, -1, self.auto_hidden)
        out = self.pos(out)
        out = out.reshape(1, batch_size, -1) # (1, batch_size, feature_size)
        out = self.cell(out)  
        out = out.reshape(batch_size, -1) # (batch_size, hidden_dim)
        out = self.linear(out)            # out shape: (batch_size, 1)
        
        return out, decode

           
def train(lr=0.01, feature_size=8, hidden_dim=32, num_layers=1, nhead=8, weight_decay=0.0, EPOCH=1000, seed=0, 
         alpha=0.0, noise_level=0.0, dropout=0.0, metric='re', is_load_weights=True):
    score_list, result_list = [], []
    setup_seed(seed)
    writer = SummaryWriter('Data/Log/Transformer15')
    for i in range(84):
        name = Battery_list[i]
        window_size = feature_size
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size) #size:train_x=17,train_y=151,train_data=17,test_data=151
        np.savetxt('Data/Transformer_15/train_data_main_15_' + str(i) + '.csv', train_data, delimiter=',')
        np.savetxt('Data/Transformer_15/test_data_main_15_' + str(i) + '.csv', test_data, delimiter=',')
        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        model = Net(feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout,
                    noise_level=noise_level)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_ = [], []
        rpe, rmse, mae = 0, 0, 0
        score_, score = [],[]
        model_ = None
        
        for epoch in tqdm(range(EPOCH)):
            X = np.reshape(train_x/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32) # (batch_size, seq_len, input_size)
            y = np.reshape(train_y[:,-1]/Rated_Capacity,(-1,1)).astype(np.float32)          # (batch_size, 1)
            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            output, decode = model(X)
            output = output.reshape(-1, 1)
            loss = criterion(output, y) + alpha * criterion(decode, X.reshape(-1, feature_size))
            loss *= 1000
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            if (epoch + 1)%10 == 0:
                test_x = train_data.copy()     #let test_x as train data
                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data): #0~[161:-1]
                    x = np.reshape(np.array(test_x[-feature_size:])/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device)   # shape (batch_size,feature_size=1,input_size)
                    pred, _ = model(x)                   # pred shape: (batch_size=1, feature_size=1)
                    next_point = pred.data.cpu().numpy()[0,0] * Rated_Capacity
                    test_x.append(next_point)      # The test values are added to the original sequence to continue to predict the next point
                    point_list.append(next_point)  # Saves the predicted value of the last point in the output sequence
                y_.append(point_list)              # Save all the predicted values
                #print(len(y_))

                loss_list.append(loss)
                rpe = relative_positional_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity*0.8)
                rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                mae = mean_absolute_error(test_data, y_[-1])
                print('Epoch:{:<2d} | loss:{:<6.4f} | RPE:{:<6.4f} | RMSE:{:<6.4f} | MAE:{:<6.4f}'.format(epoch, loss, rpe, rmse, mae))
            if metric == 'rpe':
                score = [rpe]
            elif metric == 'rmse':
                score = [rmse]
            elif metric == 'mae':
                score =[mae]
            else:
                score = [rpe, rmse, mae]
            #if (loss < 1e-1) and (score_[0] < score[0]):
                #break
            if (loss < 1e-1) and (score_[0] < score[0]) and (score_[1] < score[1]):
                break
            else:
                score_ = score.copy()
                model_ = copy.deepcopy(model)
                # save model
                torch.save(model_.state_dict(), 'Data/Transformer_15/Transformer_15_'+str(i)+'.pth')
            
            
            writer.add_scalars('Battery'+ str(i), {"Train Loss": loss,
                                                "Relative Position Error": rpe,
                                                "Root Mean Squared Error": rmse,
                                                "Mean Absolute Error": mae}, epoch)
            
        print('------------------------------------------------------------------')
        print('Battery_list:{:<2d} | loss:{:<6.4f} | RPE:{:<6.4f} | RMSE:{:<6.4f} | MAE:{:<6.4f}'.format(i, loss, rpe, rmse, mae))
        np.savetxt('Data/Transformer_15/predict_list_main_15_' + str(i) + '.csv', y_, delimiter=',')   
        with open('Data/Transformer_15/predict_list_main_15_' + str(i) + '.pkl', 'wb') as fp:
            pickle.dump(y_, fp)
        score_list.append(score_)
        result_list.append(y_[-2])
    writer.close()
    torch.save(model_, 'Data/Transformer15.pkl')
    return score_list, result_list, y_, loss_list




Rated_Capacity = 1.1
window_size = 16
feature_size = window_size
dropout = 0.0
EPOCH = 10000
nhead = 8
hidden_dim = 64
num_layers = 1
lr = 0.0043    # learning rate
weight_decay = 0.0
noise_level = 0.0
alpha = 1e-5
is_load_weights = True
metric = 'error'
seed = 0


SCORE_rpe, SCORE_rmse, SCORE_mae  = [], [], []
print('seed:{}'.format(seed))
score_list, result_list, predict_list, loss_list= train(lr=lr, feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, 
                      weight_decay=weight_decay, EPOCH=EPOCH, seed=seed, dropout=dropout, alpha=alpha, 
                      noise_level=noise_level, metric=metric, is_load_weights=is_load_weights)

with open('Data/Transformer_15/predict_list_main_15.pkl', 'wb') as fp:
            pickle.dump(result_list, fp)
print(np.array(score_list))
for s in score_list:
    SCORE_rpe.append(s[0])
    SCORE_rmse.append(s[1])
    SCORE_mae.append(s[2])
print('------------------------------------------------------------------')
#print(metric + ' mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))
print('rpe mean:{:<6.4f} | rmse mean:{:<6.4f} | mae mean:{:<6.4f}'.format(np.mean(np.array(SCORE_rpe)), np.mean(np.array(SCORE_rmse)), np.mean(np.array(SCORE_mae))))
# Calculate the means
mean_rpe = np.mean(np.array(SCORE_rpe))
mean_rmse = np.mean(np.array(SCORE_rmse))
mean_mae = np.mean(np.array(SCORE_mae))

# Stack the means into a single array
mean_values = np.array([mean_rpe, mean_rmse, mean_mae])

# Save the means to a CSV file
np.savetxt('Data/Transformer_15/Transformer_15_Score.csv', mean_values, delimiter=',')

'''
Rated_Capacity = 1.1
window_size = 16
feature_size = window_size
dropout = 0.0
EPOCH = 2000
nhead = 8
is_load_weights = False

weight_decay = 0.0
noise_level = 0.0
alpha = 0.0
metric = 'rmse'

states = {}
for lr in [1e-3, 1e-2]:
    for num_layers in [1, 2]:
        for hidden_dim in [16, 32]:
            for alpha in [1e-5, 1e-4]:
                show_str = 'lr={}, num_layers={}, hidden_dim={}, alpha={}'.format(lr, num_layers, hidden_dim, alpha)
                print(show_str)
                SCORE = []
                for seed in range(5):
                    print('seed:{}'.format(seed))
                    score_list, _, predict_list, loss_list= train(lr=lr, feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, 
                                                            weight_decay=weight_decay, EPOCH=EPOCH, seed=seed, dropout=dropout, alpha=alpha, 
                                                            noise_level=noise_level, metric=metric, is_load_weights=is_load_weights)
                    print(np.array(score_list))
                    print(metric + ': {:<6.4f}'.format(np.mean(np.array(score_list))))
                    print('------------------------------------------------------------------')
                    for s in score_list:
                        SCORE.append(s)

                print(metric + ' mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))
                states[show_str] = np.mean(np.array(SCORE))
                print('===================================================================')

min_key = min(states, key = states.get)
print('optimal parameters: {}, result: {}'.format(min_key, states[min_key]))
'''
