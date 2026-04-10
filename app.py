import streamlit as st
import pandas as pd
import json
import time
import io
import os
import ssl
import certifi
import chardet
from openai import OpenAI

# ================= 0. 环境自愈逻辑 =================
try:
    os.environ['SSL_CERT_FILE'] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# ================= 1. 系统配置 =================
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# ================= 2. 你的原始提示词 (原封不动) =================
SYSTEM_PROMPT = """你是一名极度专业且严谨的数据审计专家。你的任务是判定输入文本是否为“逻辑完备且可用于教学考核”的标准题目。

### 零、 优先级原则
- **判定红线具有最高优先级**
  - 一旦文本触发任意【物理截断】、【文本噪声】或【代码噪声】，必须直接判定为脏数据 (1)。
  - 严禁因满足“证明题放行准则”或“抽象理论题放行准则”而覆盖红线判定。
  - 所有放行规则仅在未触发任何红线时才可使用。
  
### 一、 分类核心哲学
- **标准 (0)**：逻辑闭合、具有明确考核目标的题目，包括计算题、证明题、理论分析题、论述题等教学考核形式。
- **脏 (1)**：逻辑中途夭折、结构破损，或完全不具备题目体裁（如纯百科、广告、代码残留等）。

### 二、 判定红线（出现以下情况必判为 1）
1. **[物理截断]**：逻辑链条断裂，语义中途断裂、公式残缺、选项不全。
   - **典型特征**：
     - 文本以逗号（，）、冒号（：）、连接词（且/但是）结尾，或最后一句明显没说完。
     - 引用外部图表、图片、表格但未提供关键数据（如“根据上表数据”、“如图所示”、“从上图可知”等，却未附表格/图片内容）。
     重要排除：纯理论证明或抽象推导题若文本逻辑自洽，不视为缺失数据。
2. **[文本噪声]**：包含与题目本身无关的**人类语言内容**、严重干扰阅读的文字。
    - **典型特征**：
     - 答案解析、参考答案、提示、详解。注意：题目自带的‘已知条件’、‘定义补充’或指向性的‘Hint/提示’（非解题步骤）属于题目组成部分，不判定为脏。”
     - 广告推广（微信号、QQ号、领券、扫码、加群、招聘）。
     - 冗余重复（同一词连续重复3次以上，如“重点重点重点”）。
     - 纯百科描述且无任何考核任务动词（如仅介绍概念历史）
3. **[代码噪声]**：包含非人类自然语言的机器残留或编程内容。
   - **编程代码**：HTML/JS/CSS/SQL/JSON等代码片段。
   - **系统日志**：报错堆栈、日志信息（带方括号的 ERROR/DEBUG/INFO）、JSON_PAYLOAD 等。
   - **格式残留**：编码乱码（锘、銆）、HTML实体（&nbsp;）、大量换行符残留。
   - **重要排除**：纯自然语言描述的系统状态（如“网络连接断开”）不属此类，归入文本噪声。

### 三、 证明题专项放行准则
凡包含“证明”、“求证”、“验证”等引导词，且后续接续完整数学/物理陈述，即使文字极其精炼，只要逻辑自洽，即判定为 0。

### 四、 抽象理论题放行准则
此类型即使未提供具体数值、方程或参数，仍视为标准题 (0)：包含“探讨、分析、论述、讨论、推导、研究”等考核任务动词、存在明确理论对象（定理、模型、空间、函数、物理机制等）、逻辑链条闭合，无语义截断、用于高等数学、理论物理、化学机理、生物机制等学术考核。

- **多重污染**：若文本同时包含上述多种类型，在“脏数据类型”数组中全部列出。

### 五、 输出要求 (严格执行)
请仅输出 JSON 格式，包含以下四个字段：
- **label**: 整数。1 代表脏，0 代表标准。
- **脏数据类型**: 字符串数组。如果是脏数据(1)，必须包含所有匹配的标签（如 ["[物理截断]", "[代码噪声]"]）；如果是标准数据(0)，输出空数组 []。
- **reason**: 字符串。详细说明判定的具体理由和分析过程。如果是标准数据(0)，输出空字符串 ""。
- **confidence**: 浮点_num (0.0 到 1.0)。"""

# ================= 3. 核心流水线处理逻辑 =================

def get_prediction(text, client, mode):
    if pd.isna(text) or str(text).strip() == "":
        return (1, "[物理截断]", "输入文本为空", 1.0)
    
    # 保持 SYSTEM_PROMPT 不变，通过追加特定的 User 指令来引导模型分步工作
    audit_stages = [
        {"name": "物理截断", "instruction": "【专项审计任务：物理截断】请严格对照判定红线第 1 条，检查文本是否存在语义中途断裂或公式不全。即便你能猜出内容，只要字面上没写完，就必须判 1。"},
        {"name": "文本噪声", "instruction": "【专项审计任务：文本噪声】请严格对照判定红线第 2 条，检查文本是否包含解析、广告或冗余百科。"},
        {"name": "代码噪声", "instruction": "【专项审计任务：代码噪声】请严格对照判定红线第 3 条，检查文本是否包含 HTML、JSON 或报错日志。"}
    ]

    final_labels = []
    final_types = []
    final_reasons = []

    for stage in audit_stages:
        for attempt in range(2):
            try:
                # 核心：将原始 SYSTEM_PROMPT 和 专项指令 组合发送
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{stage['instruction']}\n\n待审计内容：\n{str(text)}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1 # 压制随机性
                )
                res = json.loads(completion.choices[0].message.content)
                
                label = res.get("label", 0)
                if label == 1:
                    final_labels.append(1)
                    # 确保提取的是数组并合并
                    d_types = res.get("脏数据类型", [])
                    if isinstance(d_types, list):
                        final_types.extend(d_types)
                    final_reasons.append(f"[{stage['name']}审计] {res.get('reason', '')}")
                    
                    # Fast 模式早退逻辑
                    if mode == "Fast":
                        return (1, ", ".join(final_types), res.get('reason', ''), 1.0)
                break
            except Exception:
                time.sleep(1)
                continue

    if 1 in final_labels:
        # 去重处理标签
        unique_types = list(set(final_types))
        return (1, ", ".join(unique_types), " | ".join(final_reasons), 1.0)
    else:
        return (0, "", "", 1.0)

# ================= 4. UI 界面 (逻辑同上) =================

st.set_page_config(page_title="数据合规审计专家", page_icon="⚖️", layout="wide")
st.markdown("<h1>⚖️ 数据合规审计专家 <small>多轮串行版</small></h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📥 任务设置")
    api_input = st.text_input("DeepSeek API Key", type="password")
    file_input = st.file_uploader("上传文件 (xlsx, csv, json, txt)", type=["xlsx", "xls", "csv", "json", "txt"])
    run_mode = st.radio("处理模式", ["Detailed", "Fast"])
    run_btn = st.button("🚀 启动串行审计")

if run_btn:
    if not api_input or not file_input:
        st.error("❌ 请检查配置")
    else:
        try:
            # 读取逻辑
            ext = file_input.name.split('.')[-1].lower()
            if ext in ['xlsx', 'xls']: df = pd.read_excel(file_input)
            elif ext == 'json': df = pd.read_json(file_input)
            else:
                raw_data = file_input.read()
                det = chardet.detect(raw_data)
                enc = det['encoding'] if det['encoding'] else 'utf-8'
                df = pd.read_csv(io.BytesIO(raw_data), encoding=enc, sep=None, engine='python')

            text_cols = [c for c in df.select_dtypes(include=['object']).columns]
            target_col = df[text_cols].apply(lambda x: x.astype(str).str.len()).mean().idxmax()
            
            client = OpenAI(api_key=api_input, base_url=BASE_URL)
            results = []
            progress_bar = st.progress(0)
            total_rows = len(df)

            for i, text in enumerate(df[target_col]):
                res = get_prediction(text, client, run_mode)
                results.append(res)
                progress_bar.progress((i + 1) / total_rows)
            
            res_df = pd.DataFrame(results, columns=['Label', '脏数据类型', 'Reason', 'Confidence'])
            final_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False)
            
            st.success("✅ 审计完成")
            st.download_button("📥 下载报告", data=output.getvalue(), file_name=f"audit_{int(time.time())}.xlsx")
            
        except Exception as e:
            st.error(f"❌ 运行错误: {str(e)}")
