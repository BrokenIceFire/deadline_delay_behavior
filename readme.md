# 作业截止期限（Deadline）与拖延行为的动力学模型
##  声明：本项目为高中生数学建模大赛比赛作品

## 思路分析
---
### 模型建构
*   **行动方程**：做作业的速度，正比于当前的焦虑程度，并且受限于剩余作业量。

    $$ \frac{dx}{dt} = \mu \cdot A(t) \cdot (1 - x(t)) $$

    *（$\mu$ 是行动转化率：焦虑转化为行动的能力）*

*   **焦虑方程**：焦虑值的变化率 = 焦虑的产生 - 焦虑的自我缓解。

    $$ 
    \frac{dA}{dt} = \frac{\alpha \cdot (1 - x(t))}{1 + \gamma(T - t)} - \beta \cdot A(t) 
    $$

    *（$\alpha$ 是对任务敏感度；$\gamma$ 是拖延患者的“时间麻木系数”；$\beta$ 是自我安慰、打游戏逃避导致焦虑下降的速率）*
---
### 数据处理
数据来源：https://github.com/shuyeit/mmpsy-data/tree/main
文件路径：code\model_data_analyze\data\数据来源.md
* **第一步处理**：1_data_preprocession.py 对语料数据进行初步提取
* **第二部处理**：2_data_preprocessing.py 对提取语料用本地大模型提取参数
* **第三步处理**：analysis.py 进行归一化并线性回归查看方程是否合理
---
### 模型代码构建
main.py 中核心代码展示
```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ----------------- 1. 定义二维动力系统 -----------------
def procrastination_model(y, t, mu, alpha, gamma, beta, T):
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
```
---
***补充:本作品使用了ai进行辅助操作，但核心代码仍为团队成员手敲***