# -*- coding: utf-8 -*-
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
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

# 配置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量
mu = []
alpha = []
beta = []
gamma = []
gad = []

g_A = []
g_gad = []

def read_function():
    """从 JSON 文件读取预处理数据 - 修复：过滤无效数据并转换类型"""
    global mu, alpha, beta, gamma
    
    file_path = r'H:\作业截止期限（Deadline）与拖延行为的动力学模型\deadline_delay_behavior\code\model_verification\preprocessing_data\output.json'
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data_preprocession = js.load(file)
    
    for i, temp in enumerate(data_preprocession):
        # 【关键修复】获取参数并过滤无效值
        temp_mu = temp.get("mu", "false")
        temp_alpha = temp.get("alpha", "false")
        temp_beta = temp.get("beta_anx", "false")
        temp_gamma = temp.get("gamma", "false")
        
        # 检查是否所有参数都有效
        if temp_mu == "false" or temp_alpha == "false" or temp_beta == "false" or temp_gamma == "false":
            continue
        
        # 【关键修复】转换为 float 类型
        try:
            mu.append(float(temp_mu))
            alpha.append(float(temp_alpha))
            beta.append(float(temp_beta))
            gamma.append(float(temp_gamma))
        except (ValueError, TypeError) as e:
            print(f"第 {i} 条数据转换失败: {e}")
            continue
    
    print("数据读取成功：mu=%d, alpha=%d, beta=%d, gamma=%d" % (len(mu), len(alpha), len(beta), len(gamma)))

def read_GAD():
    """读取 GAD 焦虑量表分数 - 修复：转换为 float"""
    global gad
    
    file_path = r'H:\作业截止期限（Deadline）与拖延行为的动力学模型\deadline_delay_behavior\code\model_verification\data\wav_select.json'
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data_preprocession = js.load(file)
    
    for i, temp in enumerate(data_preprocession):
        try:
            temp_gad = temp["gad-label"]["score"]
            gad.append(float(temp_gad))
        except (KeyError, ValueError, TypeError) as e:
            print(f"第 {i} 条 GAD 数据读取失败: {e}")
            continue
    
    print("GAD 数据读取成功：%d 条" % len(gad))

class caculate():
    def __init__(self, y0):
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.mu = 0.0
        self.y0 = y0
        self.A_results = []  # 存储所有样本的 A 值
        
    def setting(self, alpha, beta, gamma, mu):
        # 【关键修复】确保参数为 float 类型
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.mu = float(mu)

    def procrastination_model(self, y, t, mu, alpha, gamma, beta, T):
        """定义微分方程"""
        x, A = y   
        # 避免分母为零或负数
        time_factor = max(1.0 + float(gamma) * (float(T) - t), 0.01)
        time_pressure = float(alpha) * (1 - x) / time_factor
        dAdt = time_pressure - float(beta) * A           
        return [float(alpha) * (1 - x), dAdt]  # 返回 [dx/dt, dA/dt]

    def add_A(self):
        """求解微分方程并计算 A(t) 的平均值"""
        self.T = 14.0
        self.t_steps = np.linspace(0, self.T, 500)
        
        try:
            sol_diligent = odeint(self.procrastination_model, self.y0, self.t_steps, 
                                args=(self.mu, self.alpha, self.gamma, self.beta, self.T))
            A = sol_diligent[:, 1]
            
            # 【修复】计算稳定阶段的平均值（去掉前后25个点）
            if len(A) > 50:
                summary = np.mean(A[25:len(A)-25])
            else:
                summary = np.mean(A)
            
            self.A_results.append(summary)
            
        except Exception as e:
            print(f"ODE 求解失败: {e}")
            self.A_results.append(0.0)

def data_transfrom_gad(np_gad):
    """GAD 分数归一化"""
    if len(np_gad) == 0:
        return np.array([])
    data_percent1 = MinMaxScaler(feature_range=(0, 1))
    g_gad = data_percent1.fit_transform(np_gad.reshape(-1, 1)).flatten()
    return g_gad
    
def data_transfrom_A(A_values):
    """A 值归一化 - 修复：避免变量名冲突"""
    if len(A_values) == 0:
        return np.array([])
    data_percent2 = MinMaxScaler(feature_range=(0, 1))
    normalized_A = data_percent2.fit_transform(np.array(A_values).reshape(-1, 1)).flatten()
    return normalized_A

def draw(gad_values, a_values, save_path=None):
    """绘制散点图并计算相关性"""
    if len(gad_values) == 0 or len(a_values) == 0:
        print("错误：数据为空，无法绘图")
        return
    
    if len(gad_values) != len(a_values):
        print(f"警告：数据长度不一致，gad={len(gad_values)}, A={len(a_values)}")
        min_len = min(len(gad_values), len(a_values))
        gad_values = gad_values[:min_len]
        a_values = a_values[:min_len]
    
    # 计算相关性
    if len(gad_values) > 2:
        r, p = stats.pearsonr(a_values, gad_values)
    else:
        r, p = 0.0, 1.0

    # 画散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(a_values, gad_values, color='steelblue', s=50, alpha=0.7, edgecolors='black')
    
    # 添加趋势线
    if len(a_values) > 2:
        z = np.polyfit(a_values, gad_values, 1)
        p_line = np.poly1d(z)
        plt.plot(a_values, p_line(a_values), "r--", alpha=0.8, linewidth=2, label='趋势线')
    
    plt.xlabel('理论行动强度 A (归一化)', fontsize=12)
    plt.ylabel('GAD 焦虑分数 (归一化)', fontsize=12)
    plt.title(f'皮尔逊相关系数 r = {r:.3f}, p = {p:.3e}', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存：{save_path}")
    
    plt.show()
    
    return r, p

def linear_regression_analysis(X, y, feature_name="A", target_name="GAD"):
    """线性回归分析"""
    if len(X) < 3 or len(y) < 3:
        print("数据量不足，无法进行回归分析")
        return None
    
    X = X.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'coef': model.coef_[0],
        'intercept': model.intercept_
    }
    
    print("\n" + "=" * 50)
    print("线性回归分析报告")
    print("=" * 50)
    print(f"特征：{feature_name} → 目标：{target_name}")
    print(f"样本数：{len(X)} (训练集={len(X_train)}, 测试集={len(X_test)})")
    print(f"回归系数：{metrics['coef']:.6f}")
    print(f"截距：{metrics['intercept']:.6f}")
    print(f"R² = {metrics['r2']:.6f}")
    print(f"RMSE = {metrics['rmse']:.6f}")
    print(f"MAE = {metrics['mae']:.6f}")
    print("=" * 50)
    
    return model, metrics

if __name__ == "__main__":
    print("=" * 60)
    print("作业截止期限与拖延行为 - 动力学模型验证")
    print("=" * 60)
    
    # 读取数据
    read_function()
    read_GAD()
    
    # 检查数据一致性
    print(f"\n数据检查:")
    print(f"  模型参数样本数：{len(mu)}")
    print(f"  GAD 分数样本数：{len(gad)}")
    
    # 【关键修复】确保两个数组长度一致
    min_samples = min(len(mu), len(gad))
    if min_samples == 0:
        print("\n错误：数据为空，程序终止")
    else:
        print(f"  有效样本数：{min_samples}")
        
        # 截取一致的长度
        mu = mu[:min_samples]
        alpha = alpha[:min_samples]
        beta = beta[:min_samples]
        gamma = gamma[:min_samples]
        gad = gad[:min_samples]
        
        # 转换为 numpy 数组
        np_mu = np.array(mu)
        np_alpha = np.array(alpha)
        np_beta = np.array(beta)
        np_gamma = np.array(gamma)
        np_gad = np.array(gad)
        
        # 归一化 GAD 分数
        new_np_gad = data_transfrom_gad(np_gad)
        
        # 创建计算对象
        caculate1 = caculate(y0=[0.0, 0.0])
        
        # 遍历所有样本计算 A(t)
        print("\n开始计算行动强度 A(t)...")
        for i in range(len(np_mu)):
            caculate1.setting(alpha=np_alpha[i], beta=np_beta[i], 
                            gamma=np_gamma[i], mu=np_mu[i])
            caculate1.add_A()
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(np_mu)} 个样本")
        
        # 归一化 A 值
        g_A = data_transfrom_A(caculate1.A_results)
        
        print(f"\n计算完成！有效 A 值数量：{len(g_A)}")
        
        # 绘制散点图
        print("\n绘制相关性散点图...")
        r, p = draw(new_np_gad, g_A, save_path='gad_A_correlation.png')
        
        # 线性回归分析
        print("\n进行线性回归分析...")
        model, metrics = linear_regression_analysis(g_A, new_np_gad, 
                                                    feature_name="行动强度A", 
                                                    target_name="GAD焦虑分数")
        
        # 保存模型
        if model is not None:
            joblib.dump(model, 'gad_prediction_model.pkl')
            print("\n模型已保存：gad_prediction_model.pkl")
        
        # 输出统计信息
        print("\n" + "=" * 60)
        print("统计汇总")
        print("=" * 60)
        print(f"  GAD 分数范围：[{np_gad.min():.2f}, {np_gad.max():.2f}]")
        print(f"  A 值范围：[{min(caculate1.A_results):.4f}, {max(caculate1.A_results):.4f}]")
        print(f"  相关系数 r = {r:.4f}")
        print(f"  显著性 p = {p:.4e}")
        if p < 0.05:
            print("  ✓ 相关性显著 (p < 0.05)")
        else:
            print("  ✗ 相关性不显著 (p ≥ 0.05)")
        print("=" * 60)