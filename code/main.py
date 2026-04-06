import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

class DynamicModel:
    def __init__(self, mu, alpha, beta, T_deadline, y0):
        # 通用参数
        self.mu = mu   # 行动力转化率
        self.alpha = alpha # 压力敏感度
        self.beta = beta  # 焦虑自我缓解率（例如通过玩手机逃避）

        # ----------------- 2. 参数设置 -----------------
        self.T_deadline = T_deadline     # 假设作业期限为14天
        self.t_steps = np.linspace(0, T_deadline, 500) # 生成500个时间点
        self.y0 = y0         # 初始状态：进度为0，焦虑值为0
    

    # ----------------- 1. 定义二维动力系统 -----------------
    def procrastination_model(self, y, t, mu, alpha, gamma, beta, T):
        """
        y[0]: x(t) 任务完成度 (0 到 1)
        y[1]: A(t) 焦虑值/压力值
        """
        x, A = y

        # 限制 x 最大为 1 (作业写完了就不写了)
        if x >= 1.0:
            dxdt = 0.0
            # 作业写完了，焦虑随着时间自然消退
            dAdt = -beta * A 
        else:
            # 1. 行动方程：进度增加速率 = 转化率 * 焦虑值 * 剩余任务量
            dxdt = mu * A * (1 - x)
            
            # 2. 焦虑方程：焦虑增加 = 剩余任务带来的时间压力 - 自我情绪缓解
            # 使用 1 + gamma*(T-t) 来体现“时间折扣”（拖延症本质）
            time_pressure = (alpha * (1 - x)) / (1 + gamma * (T - t))
            dAdt = time_pressure - beta * A
            
        return [dxdt, dAdt]

    def showImage(self):
        # 对比参数：非拖延症 vs 严重拖延症
        gamma_diligent = 0.1  # 勤奋学生：几乎不受时间远近影响
        gamma_procrast = 4.0  # 拖延学生：距离远时极度麻木，对时间极度打折

        # ----------------- 3. 微分方程数值求解 -----------------
        sol_diligent = odeint(self.procrastination_model, self.y0, self.t_steps, 
                            args=(self.mu, self.alpha, gamma_diligent, self.beta, self.T_deadline))
        sol_procrast = odeint(self.procrastination_model, self.y0, self.t_steps, 
                            args=(self.mu, self.alpha, gamma_procrast, self.beta, self.T_deadline))

        x_diligent, A_diligent = sol_diligent[:, 0], sol_diligent[:, 1]
        x_procrast, A_procrast = sol_procrast[:, 0], sol_procrast[:, 1]

        # ----------------- 4. 绘图展示 -----------------
        # 开启正常显示中文的设置
        plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows用SimHei，Mac可用Arial Unicode MS
        plt.rcParams['axes.unicode_minus'] = False 

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # 图1：任务完成度对比
        ax1.plot(self.t_steps, x_diligent, 'b-', linewidth=2.5, label='勤奋型 (低拖延系数)')
        ax1.plot(self.t_steps, x_procrast, 'r-', linewidth=2.5, label='拖延型 (高拖延系数)')
        ax1.axhline(y=1.0, color='gray', linestyle='--')
        ax1.axvline(x=self.T_deadline, color='k', linestyle=':', label='Deadline')
        ax1.set_ylabel('作业完成进度 x(t)', fontsize=12)
        ax1.set_title('作业进度动态变化曲线', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 图2：焦虑值对比
        ax2.plot(self.t_steps, A_diligent, 'b-', linewidth=2.5, label='勤奋型焦虑值')
        ax2.plot(self.t_steps, A_procrast, 'r-', linewidth=2.5, label='拖延型焦虑值')
        ax2.axvline(x=self.T_deadline, color='k', linestyle=':')
        ax2.set_xlabel('时间 t (天)', fontsize=12)
        ax2.set_ylabel('心理焦虑值 A(t)', fontsize=12)
        ax2.set_title('心理焦虑/压力动态变化曲线', fontsize=14)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = DynamicModel(mu=0.4, alpha=1.0, beta=0.8, T_deadline=60.0, y0=[0.0, 0.0])
    model.showImage()