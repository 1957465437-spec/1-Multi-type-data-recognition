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

# ================= 2. 深度审计提示词 (修复输出结构) =================
SYSTEM_PROMPT = """你是一名极度专业且严谨的数据审计专家。你的任务是判定输入文本是否为“逻辑完备且可用于教学考核”的标准题目。

### 一、 分类核心哲学
- **标准 (0)**：有明确的提问意图、完整的已知量、且逻辑闭合。
- **脏 (1)**：逻辑中途夭折、包含答案解析、或是非题目体裁（如百科、纯代码块）。

### 二、 判定红线（出现以下情况必判为 1）
1. **逻辑空洞（[物理截断]）**：语义中途断裂、公式残缺、选项不全。
2. **纯粹废料与泄露（[文本噪声]）**：包含答案解析、仅有百科/新闻背景而无提问、严重干扰阅读的文字。
3. **格式污染（[代码噪声]）**：包含代码块（如 Python 函数）、HTML/JSON 等机器残留。

### 三、 输出要求 (严格执行)
请仅输出 JSON 格式，包含以下四个字段：
- **label**: 整数。1 代表脏，0 代表标准。
- **脏数据类型**: 字符串数组。如果是脏数据(1)，必须包含所有匹配的标签（如 ["[物理截断]", "[代码噪声]"]）；如果是标准数据(0)，输出空数组 []。
- **reason**: 字符串。详细说明判定的具体理由和分析过程。如果是标准数据(0)，输出空字符串 ""。
- **confidence**: 浮点数 (0.0 到 1.0)。"""

# ================= 3. 核心处理逻辑 =================

def get_prediction(text, client, mode):
    if pd.isna(text) or str(text).strip() == "":
        return (1, "[物理截断]", "文本为空", 1.0) if mode == "Detailed" else (1, None, None, None)
    
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请审计以下内容：\n{str(text)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            res = json.loads(completion.choices[0].message.content)
            
            label = res.get("label")
            # 修正：从 JSON 中按新定义的 key 取值
            dirty_types = res.get("脏数据类型", [])
            analysis_reason = res.get("reason", "")
            confidence = res.get("confidence", 0.0)
            
            # 逻辑：标签为0时，清空类型和分析原因（保持表格整洁）
            if label == 0:
                dirty_str = ""
                reason_str = ""
            else:
                # 将数组转为逗号分隔的字符串
                dirty_str = ", ".join(str(r) for r in dirty_types) if isinstance(dirty_types, list) else str(dirty_types)
                reason_str = str(analysis_reason)
            
            return (label, dirty_str, reason_str, confidence) if mode == "Detailed" else (label, None, None, None)
                
        except Exception:
            if attempt < 2:
                time.sleep(1.0)
                continue
            return ("Error", "API异常", "API异常", 0.0)

# ================= 4. UI 界面 =================

st.set_page_config(page_title="数据合规审计专家", page_icon="⚖️", layout="wide")

st.markdown("<h1>⚖️ 数据合规审计专家</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📥 任务设置")
    api_input = st.text_input("DeepSeek API Key", type="password")
    file_input = st.file_uploader("上传文件 (xlsx, csv, json, txt)", type=["xlsx", "xls", "csv", "json", "txt"])
    run_mode = st.radio("处理模式", ["Detailed", "Fast"])
    run_btn = st.button("🚀 启动智能审计")

if run_btn:
    if not api_input or not file_input:
        st.error("❌ 请检查配置")
    else:
        try:
            ext = file_input.name.split('.')[-1].lower()
            df = pd.DataFrame()

            if ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_input)
            elif ext == 'json':
                df = pd.read_json(file_input)
            elif ext in ['csv', 'txt']:
                raw_data = file_input.read()
                if raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
                    enc = 'utf-16'
                elif raw_data.startswith(b'\xef\xbb\xbf'):
                    enc = 'utf-8-sig'
                else:
                    det = chardet.detect(raw_data)
                    enc = det['encoding'] if det['encoding'] else 'utf-8'
                
                try:
                    df = pd.read_csv(io.BytesIO(raw_data), encoding=enc, sep=None, engine='python')
                except:
                    df = pd.read_csv(io.BytesIO(raw_data), encoding='gb18030', sep=None, engine='python')

            text_cols = [c for c in df.select_dtypes(include=['object']).columns]
            if not text_cols:
                st.error("无法识别文本列")
                st.stop()
                
            target_col = df[text_cols].apply(lambda x: x.astype(str).str.len()).mean().idxmax()
            
            client = OpenAI(api_key=api_input, base_url=BASE_URL)
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, text in enumerate(df[target_col]):
                res = get_prediction(text, client, run_mode)
                results.append(res)
                progress_bar.progress((i + 1) / len(df))
                status_text.text(f"正在审计第 {i+1} 条数据...")
            
            if run_mode == "Detailed":
                # 这里对应函数返回的四个值
                res_df = pd.DataFrame(results, columns=['Label', '脏数据类型', 'Reason', 'Confidence'])
            else:
                res_df = pd.DataFrame([r[0] for r in results], columns=['Label'])
            
            final_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False)
            
            st.success(f"✅ 审计完成！分析列：【{target_col}】")
            st.download_button("📥 下载审计报告", data=output.getvalue(), file_name="audit_report.xlsx")
            
        except Exception as e:
            st.error(f"出错: {e}")
