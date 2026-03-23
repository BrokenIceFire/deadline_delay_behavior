import json as js
import os
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import joblib

mu = []
alpha = []
beta = []
gamma = []

g_mu = []
g_alpha = []
g_gamma = []
g_beta = []

def read_function():
    with open('H:\作业截止期限（Deadline）与拖延行为的动力学模型\code\model_data_analyze\data_preprocession.json', 'r', encoding='utf-8') as file:
        data_preprocession = js.load(file)
    for i,temp in enumerate(data_preprocession):
        temp_alpha=0
        temp_beta=0
        temp_mu=0
        temp_gamma=0
        
        temp_mu = temp.get("mu","false")
        if temp_mu == "false":
            continue
        else:         
            mu.append(temp_mu)
            
        temp_alpha = temp.get("alpha","false")
        if temp_alpha == "false":
            continue
        else:         
            alpha.append(temp_alpha)
            
        temp_beta = temp.get("beta","false")
        if temp_beta == "false":
            continue
        else:         
            beta.append(temp_beta)
        
        temp_gamma = temp.get("gamma","false")
        if temp_gamma == "false":
            continue
        else:         
            gamma.append(temp_gamma)
    print(mu,alpha,beta,gamma)


def data_transfrom(np_mu,np_alpha,np_beta,np_gamma):
    #g_mu取值在[0.1,0.5]之间
    data_percent0 = MinMaxScaler(feature_range=(0.1, 0.5))
    g_mu = data_percent0.fit_transform(np_mu.reshape(-1, 1))
    
    data_percent1 = MinMaxScaler(feature_range=(0.01, 5.0))
    g_gamma = data_percent1.fit_transform(np_gamma.reshape(-1, 1))
    
    data_percent2 = MinMaxScaler(feature_range=(0.5, 2.0))
    g_alpha = data_percent1.fit_transform(np_alpha.reshape(-1, 1))
    
    data_percent1 = MinMaxScaler(feature_range=(0.1, 1.0))
    g_beta = data_percent1.fit_transform(np_beta.reshape(-1, 1))
    
    print("\n",g_mu,"\n",g_gamma,"\n",g_alpha,"\n",g_beta)
    
if __name__ == "__main__":
    read_function()
    np_mu = np.array(mu)
    np_alpha = np.array(alpha)
    np_beta = np.array(beta)
    np_gamma = np.array(gamma)
    data_transfrom(np_mu,np_alpha,np_beta,np_gamma)