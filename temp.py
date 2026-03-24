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
gad = []

g_mu = []
g_alpha = []
g_gamma = []
g_beta = []
g_gad = []

def read_function():
    with open(r'C:\Users\Administrator\Documents\deadline_delay_behavior\code\model_data_analyze\preprocessing_data\output.json', 'r', encoding='utf-8') as file:
        data_preprocession = js.load(file)
    for i,temp in enumerate(data_preprocession):
        temp_alpha=0
        temp_beta=0
        temp_mu=0
        temp_gamma=0
        
        temp_mu = temp.get("mu","false")
        mu.append(temp_mu)
            
        temp_alpha = temp.get("alpha","false")
        alpha.append(temp_alpha)
            
        temp_beta = temp.get("beta_anx","false")
        beta.append(temp_beta)
        
        temp_gamma = temp.get("gamma","false")      
        gamma.append(temp_gamma)
    # print(mu,alpha,beta,gamma)

def read_GAD():
    with open(r'C:\Users\Administrator\Documents\deadline_delay_behavior\code\model_data_analyze\data\wav_select.json', 'r', encoding='utf-8') as file:
        data_preprocession = js.load(file)
    for i,temp in enumerate(data_preprocession):
        temp_gad = temp["gad-label"]["score"]
        gad.append(temp_gad)


def data_transfrom(np_mu,np_alpha,np_beta,np_gamma,np_gad):
    #g_mu取值在[0.1,0.5]之间
    data_percent0 = MinMaxScaler(feature_range=(0.1, 0.5))
    g_mu = data_percent0.fit_transform(np_mu.reshape(-1, 1))
    
    data_percent1 = MinMaxScaler(feature_range=(0.01, 5.0))
    g_gamma = data_percent1.fit_transform(np_gamma.reshape(-1, 1))
    
    data_percent2 = MinMaxScaler(feature_range=(0.5, 2.0))
    g_alpha = data_percent2.fit_transform(np_alpha.reshape(-1, 1))
    
    data_percent3 = MinMaxScaler(feature_range=(0.1, 1.0))
    g_beta = data_percent3.fit_transform(np_beta.reshape(-1, 1))
    
    data_percent4 = MinMaxScaler(feature_range=(0.1, 27))
    g_gad = data_percent4.fit_transform(np_gad.reshape(-1, 1))

    print("\n",g_mu,"\n",g_gamma,"\n",g_alpha,"\n",g_beta,"\n",g_gad)

def kafang(ga,gb,gay,gg,gm):
    # # 1. 行动方程：进度增加速率 = 转化率 * 焦虑值 * 剩余任务量
    # dxdt = mu * A * (1 - x)
    
    # # 2. 焦虑方程：焦虑增加 = 剩余任务带来的时间压力 - 自我情绪缓解
    # # 使用 1 + gamma*(T-t) 来体现“时间折扣”（拖延症本质）
    # time_pressure = (alpha * (1 - x)) / (1 + gamma * (T - t))
    # dAdt = time_pressure - beta * A
    pass

if __name__ == "__main__":
    read_function()
    read_GAD()
    np_mu = np.array(mu)
    np_alpha = np.array(alpha)
    np_beta = np.array(beta)
    np_gamma = np.array(gamma)
    np_gad = np.array(gad)
    data_transfrom(np_mu,np_alpha,np_beta,np_gamma,np_gad)
    kafang(g_alpha,g_beta,g_gad,g_gamma,g_mu)