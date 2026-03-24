import ollama
import json
import re
import numpy as np
import json
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def extract_psychological_metrics(text_list, model_name="qwen2"):
    """
    第一阶段：利用 LLM 提取心理学指标 (Sa, Sn, Se, Sc)
    """
    # 1. 将零碎的语料按时间顺序拼接成一段完整的上下文
    combined_text = "\n".join([f"发言片段 {i+1}: {t}" for i, t in enumerate(text_list)])
    
    # 2. 构造强大的系统提示词 (System Prompt)
    system_prompt = """
    你是一名资深的行为心理学家。请分析以下学生在面临课业压力/截止期限时的真实语音访谈记录。
    这些记录是按时间顺序发生的碎片化表达。你需要透过表象（包括逃避、防御性攻击或敷衍），评估该学生在以下四个维度的心理学得分。
    
    得分要求：[0.00, 1.00] 的浮点数。
    
    维度定义：
    1. Sa (Sensitivity_Score / 任务敏感度): 对未完成任务的心理压力感和焦虑底色（0代表毫无波澜，1代表极度敏感焦虑）。
    2. Sn (Numbness_Score / 时间麻木感): 对期限远近的迟钝程度（0代表极度在意时间，1代表死猪不怕开水烫，不到最后绝不着急）。
    3. Se (Escape_Score / 逃避倾向): 通过娱乐、闲聊或情绪发泄（如攻击性言语）来缓解压力的倾向（0代表直面任务，1代表极度爱逃避）。
    4. Sc (Control_Score / 自我控制力): 将焦虑转化为实际行动的自控和专注能力（0代表彻底摆烂/失控，1代表极度理性自律）。

    【重要限制】：请严格只输出一个 JSON 格式的字典，不要包含任何前言后语，不要包含 Markdown 格式！
    输出模板：
    {"Sa": 0.85, "Sn": 0.40, "Se": 0.70, "Sc": 0.20}
    """

    print(f"正在调用本地大模型 [{model_name}] 进行心理画像分析，请稍候...")
    
    try:
        # 3. 发送请求给本地 Ollama
        response = ollama.chat(model=model_name, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"学生的语料如下：\n{combined_text}"}
        ])
        
        # 4. 获取 AI 的原始回复
        raw_output = response['message']['content']
        
        # 5. 防错机制：用正则表达式强行抠出 JSON 括号里面的内容
        # 这样即使模型回复 "好的，这是结果：{...}" 也能成功提取
        match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            metrics = json.loads(json_str)
            return metrics
        else:
            print("解析失败，未找到 JSON 格式数据。AI 原始回复为：\n", raw_output)
            return {"Sa": 0.5, "Sn": 0.5, "Se": 0.5, "Sc": 0.5} # 容错默认值
            
    except Exception as e:
        print(f"调用 Ollama 时发生错误: {e}")
        print("请检查 Ollama 是否已启动，以及模型名称是否拼写正确。")
        return {"Sa": 0.5, "Sn": 0.5, "Se": 0.5, "Sc": 0.5}


def map_parameters(Sa, Sn, Se, Sc, params=None):
    """
    将大模型输出的心理特质得分映射为微分方程参数。
    
    参数:
        Sa, Sn, Se, Sc : float or array-like, 取值范围 [0,1]
        params : dict, 可选, 覆盖默认基准参数
    返回:
        mu, alpha, gamma, beta_anx : 四个模型参数
    """
    # 默认基准参数 (基于文献建议)
    default_params = {
        # μ 相关
        'mu_base': 0.5,      # 基准行动速率 (1/最短完成天数)
        'lambda_anx': 3.0,   # 焦虑抑制系数 (Sa对μ的负向影响)
        # α 相关
        'alpha_max': 2.0,    # 最大敏感度
        'k_alpha': 10.0,     # 敏感度S形曲线的陡峭度
        # γ 相关
        'gamma_max': 0.2,    # 最大麻木系数 (对应最短唤醒距离5天)
        'm_gamma': 10.0,     # 麻木感S形曲线的陡峭度
        # β (焦虑缓解率) 相关
        'beta0': 0.4,        # 基准缓解率
        'beta1': 1.0,        # 逃避倾向的边际效应
        'beta2': 0.5         # 自控力与逃避的交互效应
    }
    
    if params is not None:
        default_params.update(params)
    
    # 转换为numpy数组以便向量化处理
    Sa = np.asarray(Sa)
    Sn = np.asarray(Sn)
    Se = np.asarray(Se)
    Sc = np.asarray(Sc)
    
    # 1. 行动转化率 μ
    # μ = μ_base * (1 / (1 + λ * Sa)) * Sc
    mu = default_params['mu_base'] * (1 / (1 + default_params['lambda_anx'] * Sa)) * Sc
    
    # 2. 任务敏感度 α
    # 将 Sa 映射到 β-δ 模型中的 β 参数 (当下偏好)
    beta_param = 1 / (1 + np.exp(default_params['k_alpha'] * (Sa - 0.5)))
    alpha = default_params['alpha_max'] * (1 - beta_param)
    
    # 3. 时间麻木系数 γ
    # 将 Sn 映射到 δ 参数 (耐心)
    delta_param = 1 / (1 + np.exp(default_params['m_gamma'] * (Sn - 0.5)))
    gamma = default_params['gamma_max'] * (1 - delta_param)
    
    # 4. 焦虑缓解率 β_anx (为避免与β参数混淆，命名为beta_anx)
    beta_anx = default_params['beta0'] + default_params['beta1'] * Se + default_params['beta2'] * (Se * Sc)
    
    return mu, alpha, gamma, beta_anx


def student_parameters(Sa, Sn, Se, Sc, params=None):
    """返回每个学生的四个参数 (支持单个值或数组)"""
    mu, alpha, gamma, beta_anx = map_parameters(Sa, Sn, Se, Sc, params)
    mu = round(float(mu), 3)
    alpha = round(float(alpha), 3)
    gamma = round(float(gamma), 3)
    beta_anx = round(float(beta_anx), 3)
    return {
        'mu': mu,
        'alpha': alpha,
        'gamma': gamma,
        'beta_anx': beta_anx
    }


if __name__ == "__main__":

    data = json.load(open('wav_select.json', 'r', encoding='utf-8'))

    # data = [
    # {
    #     "user_id": "user_1268",
    #     "phq-label": {"score": 10, "level": "Moderate"},
    #     "gad-label": {"score": 2, "level": "Normal"},
    #     "audios": {
    #         "wav_1.wav": "没有",
    #         "wav_2.wav": "之前学习舞蹈很快乐，现在因为舞蹈影响了我的",
    #         "wav_3.wav": "好好好",
    #         "wav_4.wav": "我一般都是星期六写作业，然后星期六晚上出去",
    #         "wav_5.wav": "遇到一些令你伤心痛苦的事情，就不要再一直去",
    #         "wav_6.wav": "嗯，开心",
    #         "wav_7.wav": "劳逸结合，适当放松，"
    #     }
    # },
    # {
    #     "user_id": "user_1269",
    #     "phq-label": {"score": 9, "level": "Mild"},
    #     "gad-label": {"score": 17, "level": "Moderate"},
    #     "audios": {
    #         "wav_1.wav": "然后然后还有人意的想，我跟他约好了，到时",
    #         "wav_2.wav": "这么快啊，保持活力，希望有什么秘诀吗？没有",
    #         "wav_3.wav": "我最近感到压抑和难过的事吗？我体育并不怎么",
    #         "wav_4.wav": "呃呃，年级第一次次考。呃，呃，然后梦梦想是",
    #         "wav_5.wav": "嗯，这种状态给我的，我操，吓死呃状态学习这",
    #         "wav_6.wav": "现在的学习压力特别好，还有我呃我现在不怎么",
    #         "wav_7.wav": "我，最近我心里面总是很慌，因为旁边有个人又",
    #         "wav_8.wav": "然后感觉当然是这样说，每天早上的时候觉得处",
    #         "wav_9.wav": "空余的时间我喜欢看些小说，有本书叫做某某当"
    #     }
    # }
    # ]

    # 构建二维列表：每个子列表是一个用户的全部对话文本（按 wav 编号排序）
    student_texts = []

    for user in data:
        audios_dict = user.get("audios", {})
        # 按 wav_X.wav 的数字部分排序
        sorted_keys = sorted(audios_dict.keys(), key=lambda x: int(x.split('_')[1].split('.')[0]))
        user_dialogues = [audios_dict[key] for key in sorted_keys]
        student_texts.append(user_dialogues)

    # 输出结果
    print(student_texts)

    t = 1
    params = []
    tmp = []
    for student_text in student_texts:
        print(f"正在处理第{t}/{len(student_texts)}位学生的对话文本...")
        for j in range(1, 4):
            llm_scores = extract_psychological_metrics(student_text, model_name="deepseek-r1:8b")
            mu, alpha, gamma, beta_anx = map_parameters(llm_scores['Sa'], llm_scores['Sn'], llm_scores['Se'], llm_scores['Sc'])
            dict_params = student_parameters(llm_scores['Sa'], llm_scores['Sn'], llm_scores['Se'], llm_scores['Sc'])

            print(f"第{j}次迭代,共3次: {dict_params}")
            tmp.append(dict_params)
        avg_dict = {key: round(sum(d[key] for d in tmp) / len(tmp), 3) for key in tmp[0].keys()}
        print(avg_dict)

        params.append(avg_dict)
        t += 1
            
    
    print(params)
    # 保存为JSON文件
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("已成功保存为 output.json")
