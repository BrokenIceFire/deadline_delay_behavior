# -*- coding: utf-8 -*-
import json as js
import os
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

# 抑制 matplotlib 字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 配置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量
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
    """从 JSON 文件读取预处理数据"""
    global mu, alpha, beta, gamma
    
    file_path = r'C:\Users\Administrator\Documents\deadline_delay_behavior\code\model_data_analyze\preprocessing_data\output.json'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data_preprocession = js.load(file)
        
        for i, temp in enumerate(data_preprocession):
            temp_mu = temp.get("mu", "false")
            if temp_mu == "false":
                continue
            else:         
                mu.append(temp_mu)
                
            temp_alpha = temp.get("alpha", "false")
            if temp_alpha == "false":
                continue
            else:         
                alpha.append(temp_alpha)
                
            temp_beta = temp.get("beta", "false")
            if temp_beta == "false":
                continue
            else:         
                beta.append(temp_beta)
            
            temp_gamma = temp.get("gamma", "false")
            if temp_gamma == "false":
                continue
            else:         
                gamma.append(temp_gamma)
        
        print("数据读取成功：mu=%d, alpha=%d, beta=%d, gamma=%d" % (len(mu), len(alpha), len(beta), len(gamma)))
        
    except FileNotFoundError:
        print("文件未找到：%s" % file_path)
    except Exception as e:
        print("读取错误：%s" % str(e))
def read_GAD():
    with open(r'C:\Users\Administrator\Documents\deadline_delay_behavior\code\model_data_analyze\data\wav_select.json', 'r', encoding='utf-8') as file:
        data_preprocession = js.load(file)
    for i,temp in enumerate(data_preprocession):
        temp_gad = temp["gad-label"]["score"]
        gad.append(temp_gad)

def data_transform(np_mu, np_alpha, np_beta, np_gamma):
    """数据归一化转换函数"""
    global g_mu, g_alpha, g_beta, g_gamma
    
    # 对 mu 进行归一化
    if len(np_mu) > 0:
        data_percent0 = MinMaxScaler(feature_range=(0.1, 0.5))
        g_mu = data_percent0.fit_transform(np_mu.reshape(-1, 1)).flatten()
        print("  g_mu:   [%.4f, %.4f]" % (g_mu.min(), g_mu.max()))
    
    # 对 gamma 进行归一化
    if len(np_gamma) > 0:
        data_percent1 = MinMaxScaler(feature_range=(0.01, 5.0))
        g_gamma = data_percent1.fit_transform(np_gamma.reshape(-1, 1)).flatten()
        print("  g_gamma:[%.4f, %.4f]" % (g_gamma.min(), g_gamma.max()))
    
    # 对 alpha 进行归一化 (修正了原代码中 feature_range 最小值大于最大值的问题)
    if len(np_alpha) > 0:
        data_percent2 = MinMaxScaler(feature_range=(0.0, 0.5))
        g_alpha = data_percent2.fit_transform(np_alpha.reshape(-1, 1)).flatten()
        print("  g_alpha:[%.4f, %.4f]" % (g_alpha.min(), g_alpha.max()))
    
    # 对 beta 进行归一化
    if len(np_beta) > 0:
        data_percent3 = MinMaxScaler(feature_range=(0.1, 1.0))
        g_beta = data_percent3.fit_transform(np_beta.reshape(-1, 1)).flatten()
        print("  g_beta: [%.4f, %.4f]" % (g_beta.min(), g_beta.max()))
    
        data_percent4 = MinMaxScaler(feature_range=(0.1, 27))
        g_gad = data_percent4.fit_transform(np_gad.reshape(-1, 1))
    print("归一化完成:")
    return True

def linear_regression(X, y, feature_names=None):
    """线性回归建模函数"""
    test_size = 0.2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print("数据集划分：训练集=%d, 测试集=%d" % (len(X_train), len(X_test)))
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }
    
    return model, metrics

def print_model_report(model, metrics, feature_names=None, param_name=""):
    """打印模型报告"""
    print("\n" + "=" * 60)
    print("线性回归模型报告 - %s" % param_name)
    print("=" * 60)
    
    print("\n[模型参数]")
    print("  截距 (b): %.6f" % metrics['intercept'])
    
    if feature_names is None:
        feature_names = ["特征%d" % (i+1) for i in range(len(metrics['coefficients']))]
    
    print("\n[特征系数]")
    for i, (name, coef) in enumerate(zip(feature_names, metrics['coefficients'])):
        print("  %s (w%d): %.6f" % (name, i, coef))
    
    print("\n[训练集评估]")
    print("  R^2 分数：%.6f" % metrics['train_r2'])
    print("  RMSE: %.6f" % metrics['train_rmse'])
    print("  MAE: %.6f" % metrics['train_mae'])
    
    print("\n[测试集评估]")
    print("  R^2 分数：%.6f" % metrics['test_r2'])
    print("  RMSE: %.6f" % metrics['test_rmse'])
    print("  MAE: %.6f" % metrics['test_mae'])
    
    print("\n[模型质量评估]")
    if metrics['test_r2'] > 0.8:
        print("  优秀：模型拟合效果很好")
    elif metrics['test_r2'] > 0.6:
        print("  良好：模型拟合效果较好")
    elif metrics['test_r2'] > 0.4:
        print("  一般：模型拟合效果尚可")
    else:
        print("  较差：建议尝试其他模型或特征工程")
    
    print("=" * 60)

def plot_results(X_test, y_test, y_pred_test, X_train, y_train, y_pred_train, model, feature_name="X", param_name=""):
    """绘制回归结果图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_test, y_pred_test, color='red', alpha=0.6, label='Test', s=100, edgecolors='black')
    axes[0].scatter(y_train, y_pred_train, color='blue', alpha=0.4, label='Train', s=80, edgecolors='black')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2, label='Ideal')
    axes[0].set_xlabel('True Value')
    axes[0].set_ylabel('Predicted Value')
    axes[0].set_title('Predicted vs True - %s' % param_name)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if X_test.shape[1] == 1:
        axes[1].scatter(X_test, y_test, color='red', alpha=0.6, label='Test', s=100, edgecolors='black')
        axes[1].scatter(X_train, y_train, color='blue', alpha=0.4, label='Train', s=80, edgecolors='black')
        X_sorted = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
        y_sorted = model.predict(X_sorted)
        axes[1].plot(X_sorted, y_sorted, 'g-', linewidth=2, label='Fit Line')
        axes[1].set_xlabel(feature_name + '(Normalized)')
        axes[1].set_ylabel('Target(Normalized)')
        axes[1].set_title('Linear Regression Fit - %s' % param_name)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results_%s.png' % param_name.lower(), dpi=300, bbox_inches='tight')
    print("结果图已保存：linear_regression_results_%s.png" % param_name.lower())
    plt.show()

def analyze_single_parameter(param_data, param_name, target_data, target_name):
    """对单个参数进行线性回归分析"""
    print("\n" + "=" * 60)
    print("参数分析：%s -> %s" % (param_name, target_name))
    print("=" * 60)
    
    if len(param_data) == 0 or len(target_data) == 0:
        print("错误：%s 或 %s 数据为空" % (param_name, target_name))
        return None, None
    
    min_len = min(len(param_data), len(target_data))
    print("统一数据长度：%d 个样本" % min_len)
    
    X_single = param_data[:min_len].reshape(-1, 1)
    y_single = target_data[:min_len]
    
    model_single, metrics_single = linear_regression(X_single, y_single, feature_names=[param_name])
    
    if model_single is not None:
        print_model_report(model_single, metrics_single, feature_names=[param_name], param_name=param_name)
        
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, test_size=0.2, random_state=42)
        y_pred_train = model_single.predict(X_train)
        y_pred_test = model_single.predict(X_test)
        plot_results(X_test, y_test, y_pred_test, X_train, y_train, y_pred_train, model_single, 
                    feature_name=param_name, param_name=param_name)
        
        joblib.dump(model_single, 'linear_model_%s.pkl' % param_name.lower())
        print("\n模型已保存：linear_model_%s.pkl" % param_name.lower())
    
    return model_single, metrics_single

if __name__ == "__main__":
    print("=" * 60)
    print("作业截止期限与拖延行为 - 动力学模型线性回归分析")
    print("=" * 60)
    
    read_function()
    read_GAD()
    
    if len(mu) == 0 and len(alpha) == 0 and len(beta) == 0 and len(gamma) == 0:
        print("\n错误：数据读取失败，程序终止")
    else:
        np_mu = np.array(mu) if len(mu) > 0 else np.array([])
        np_alpha = np.array(alpha) if len(alpha) > 0 else np.array([])
        np_beta = np.array(beta) if len(beta) > 0 else np.array([])
        np_gamma = np.array(gamma) if len(gamma) > 0 else np.array([])
        np_gad = np.array(gad)
        
        print("\n数据概览:")
        if len(np_mu) > 0:
            print("  mu:    %d 个样本，范围 [%.4f, %.4f]" % (len(np_mu), np_mu.min(), np_mu.max()))
        if len(np_alpha) > 0:
            print("  alpha: %d 个样本，范围 [%.4f, %.4f]" % (len(np_alpha), np_alpha.min(), np_alpha.max()))
        if len(np_beta) > 0:
            print("  beta:  %d 个样本，范围 [%.4f, %.4f]" % (len(np_beta), np_beta.min(), np_beta.max()))
        if len(np_gamma) > 0:
            print("  gamma: %d 个样本，范围 [%.4f, %.4f]" % (len(np_gamma), np_gamma.min(), np_gamma.max()))
        
        print("\n数据归一化:")
        if not data_transform(np_mu, np_alpha, np_beta, np_gamma, np_gad):
            print("\n归一化失败，程序终止")
        else:
            # 分别对三个参数进行独立分析 (排除 gamma)
            models = {}
            metrics = {}
            
            # 分析 beta (beta -> mu)
            if len(g_beta) > 0 and len(g_mu) > 0:
                models['beta'], metrics['beta'] = analyze_single_parameter(g_beta, 'beta', g_mu, 'mu')
            
            # 分析 mu (mu -> alpha)
            if len(g_mu) > 0 and len(g_alpha) > 0:
                models['mu'], metrics['mu'] = analyze_single_parameter(g_mu, 'mu', g_alpha, 'alpha')
            
            # 分析 alpha (alpha -> beta)
            if len(g_alpha) > 0 and len(g_beta) > 0:
                models['alpha'], metrics['alpha'] = analyze_single_parameter(g_alpha, 'alpha', g_beta, 'beta')
            
            # Gamma 不进行线性回归，仅输出归一化后的最大最小值
            print("\n" + "=" * 60)
            print("Gamma 参数说明 (不进行线性回归)")
            print("=" * 60)
            if len(g_gamma) > 0:
                print("  g_gamma 归一化范围：[%.4f, %.4f]" % (g_gamma.min(), g_gamma.max()))
            else:
                print("  g_gamma 数据为空")
            print("=" * 60)
            
            # 多特征线性回归 (可选，排除 gamma)
            # 这里为了保持风格一致性，且不使用 gamma，暂不进行多特征回归，或仅使用 mu, alpha, beta 组合
            # 为简化并严格遵循"gamma 不用线性回归”，此处省略涉及 gamma 的多特征回归
            
            print("\n" + "=" * 60)
            print("分析完成！")
            print("=" * 60)
            
            # 打印汇总报告 (排除 gamma)
            print("\n" + "=" * 60)
            print("各参数模型汇总")
            print("=" * 60)
            for param_name in ['beta', 'mu', 'alpha']:
                if param_name in metrics and metrics[param_name] is not None:
                    print("\n%s: 测试集 R^2 = %.6f" % (param_name.upper(), metrics[param_name]['test_r2']))
            print("=" * 60)
            # 再次确认输出 gamma 归一化值
            if len(g_gamma) > 0:
                print("  g_gamma:[%.4f, %.4f]" % (g_gamma.min(), g_gamma.max()))
            print("=" * 60)