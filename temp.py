import json as js
import os
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import joblib
from scipy.integrate import odeint

mu = []
alpha = []
beta = []
gamma = []
gad = []

g_A = []
g_gad = []

def read_function():
    with open(r'H:\作业截止期限（Deadline）与拖延行为的动力学模型\deadline_delay_behavior\code\model_verification\data_preprocession.json', 'r', encoding='utf-8') as file:
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
    with open(r'H:\作业截止期限（Deadline）与拖延行为的动力学模型\deadline_delay_behavior\code\model_verification\data\wav_select.json', 'r', encoding='utf-8') as file:
        data_preprocession = js.load(file)
    for i,temp in enumerate(data_preprocession):
        temp_gad = temp["gad-label"]["score"]
        gad.append(temp_gad)

##规划要把这个类给拆开，把gad与A分别归一，线性回归，画出图像
class caculate():
    def __init__(self,y0):
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.mu = 0
        self.y0 = y0
        
    def setting(self,alpha,beta,gamma,mu):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        
    def procrastination_model(self, y, t, mu, alpha, gamma, beta, T):
        x, A = y   
        time_pressure = (alpha * (1 - x)) / (1 + gamma * (T - t))
        dAdt = time_pressure - beta * A           
        return [dAdt]

    def add_A(self):
        self.T = 14.0
        self.t_steps = np.linspace(0, self.T, 500)
        sol_diligent = odeint(self.procrastination_model, self.y0, self.t_steps, 
                            args=(self.mu, self.alpha, self.gamma, self.beta, self.T))
        A = sol_diligent[:, 1]
        
        summary = 0
        temp = data_transfrom_A_diligent(A)
        for i in range(25,len(temp)-25):
            summary+=temp[i]
        summary=summary/len(temp)-50
        g_A.append(summary)

def data_transfrom_gad(np_gad):
    data_percent1 = MinMaxScaler(feature_range=(0,1))
    g_gad = data_percent1.fit_transform(np_gad.reshape(-1, 1))
    return(g_gad)
    
def data_transfrom_A_diligent(A_diligent):
    data_percent2 = MinMaxScaler(feature_range=(0,1))
    g_A = data_percent2.fit_transform(A_diligent.reshape(-1, 1))
    return(g_A)

def data_transfrom_new_A(A_diligent):
    data_percent2 = MinMaxScaler(feature_range=(0,1))
    new_g_A = data_percent2.fit_transform(A_diligent.reshape(-1, 1))
    return(new_g_A)

def draw(gad,a):
    # 算相关性
    r, p = stats.pearsonr(a,gad)

    # 画散点图
    plt.scatter(a, gad, color='steelblue')
    plt.xlabel('理论值')
    plt.ylabel('实际值')
    plt.title(f'皮尔逊相关系数 r = {r:.3f}, p = {p:.3e}')
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    read_function()
    read_GAD()
    np_mu = np.array(mu)
    np_alpha = np.array(alpha)
    np_beta = np.array(beta)
    np_gamma = np.array(gamma)
    np_gad = np.array(gad)
    new_np_gad = data_transfrom_gad(np_gad)
    caculate1 = caculate(y0=[0.0, 0.0])
    
    for i in range(len(np_mu)):
        caculate1.setting(alpha=np_alpha[i],beta=np_beta[i],gamma=np_gamma[i],mu=np_mu[i])
        caculate1.add_A()
    g_A = data_transfrom_new_A(g_A)
    
    draw(new_np_gad,g_A)
        

