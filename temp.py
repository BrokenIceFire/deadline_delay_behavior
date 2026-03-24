import json as js
import os
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.integrate import odeint

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

##规划要把这个类给拆开，把gad与A分别归一，线性回归，画出图像
class DynamicModel:
    def __init__(self, mu, alpha, beta, T, y0, gamma,gv_gad):
        self.mu = mu   
        self.alpha = alpha 
        self.beta = beta  
        self.gamma = gamma
        self.T = T 
        self.t_steps = np.linspace(0, T, 500)
        self.y0 = y0
        self.gad = gv_gad
                     
    def procrastination_model(self, y, t, mu, alpha, gamma, beta, T):
        x, A = y
        
        dxdt = mu * A * (1 - x)
        time_pressure = (alpha * (1 - x)) / (1 + gamma * (T - t))
        dAdt = time_pressure - beta * A
            
        return [dxdt, dAdt]

    def data_transfrom(self,A_diligent):
        data_percent1 = MinMaxScaler(feature_range=(0,1))
        self.gad = data_percent1.fit_transform(self.gad.reshape(-1, 1))
        data_percent2 = MinMaxScaler(feature_range=(0,1))
        A_diligent = data_percent2.fit_transform(A_diligent.reshape(-1, 1))
        summary=0
        for i in range(1,len(A_diligent)-1):
            summary+=A_diligent[i]
        summary=summary/len(A_diligent)
        det_gad_A = abs(summary-gad)##有大问题，这里列表和数据读取没分开
        
    def showImage(self):
        sol_diligent = odeint(self.procrastination_model, self.y0, self.t_steps, 
                            args=(self.mu, self.alpha, self.gamma, self.beta, self.T))
        x_diligent, A_diligent = sol_diligent[:, 0], sol_diligent[:, 1]
        self.data_transfrom(A_diligent=A_diligent)


if __name__ == "__main__":
    # read_function()
    # read_GAD()
    # np_mu = np.array(mu)
    # np_alpha = np.array(alpha)
    # np_beta = np.array(beta)
    # np_gamma = np.array(gamma)
    # np_gad = np.array(gad)
    # data_transfrom(np_mu,np_alpha,np_beta,np_gamma,np_gad)
    model = DynamicModel(gamma=4.0,mu=0.4, alpha=1.0, beta=0.8, T=60.0, y0=[0.0, 0.0])
    model.showImage()