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

# ================= 2. 深度审计提示词 (原封不动) =================
SYSTEM_PROMPT = """你是一名极度专业且严谨的数据审计专家。你的任务是判定输入文本是否为“逻辑完备且可用于教学考核”的标准题目。

### 一、 分类核心哲学
- **标准 (0)**：有明确的提问意图、完整的已知量、且逻辑闭合。
- **脏 (1)**：逻辑中途夭折、包含答案解析、或是非题目体裁（如百科、纯代码块）。

### 二、 判定红线（出现以下情况必判为 1）
1. **逻辑空洞（[物理截断]）**：语义中途断裂、公式残缺、选项不全。
- **注意**：文本以逗号（，）、冒号（：）、连接词（且/但是）结尾，或最后一句明显没说完，哪怕前面说得再完整，只要最后是半句话，必报此项。
2. **文本噪声**：包含与题目本身无关的**人类语言内容**、严重干扰阅读的文字。
   - **典型特征**：
     - 答案解析、参考答案、提示、详解。
     - 广告推广（微信号、QQ号、领券、扫码、加群、招聘）。
     - 冗余重复（同一词连续重复3次以上，如“重点重点重点”）。
     - 纯百科/背景介绍，无任何提问意图。
3. **格式污染（[代码噪声]）** —— 【看字符合法性】：
   - **代码块**：包含 <script>、{json}、def main(): 等。
   - **乱码与残留**：包含 `锘跨`、`銆`、`â€` 等编码错误产生的乱码，或 `\n\t` 等大量换行符/缩进符残留。
   - **判定逻辑**：只要文本中出现了“非人类正常书写所需的特殊符号或机器残留”，必报此项。

- **多重污染**：若文本同时包含上述多种类型，在“脏数据类型”数组中全部列出。

### 三、 输出要求 (严格执行)
请仅输出 JSON 格式，包含以下四个字段：
- **label**: 整数。1 代表脏，0 代表标准。
- **脏数据类型**: 字符串数组。如果是脏数据(1)，必须包含所有匹配的标签（如 ["[物理截断]", "[代码噪声]"]）；如果是标准数据(0)，输出空数组 []。
- **reason**: 字符串。详细说明判定的具体理由和分析过程。如果是标准数据(0)，输出空字符串 ""。
- **confidence**: 浮点_num (0.0 到 1.0)。"""

# ================= 3. 核心处理逻辑 =================

def get_prediction(text, client, mode):
    # 排雷点 1：增强空值处理，确保返回 4 个元素
    if pd.isna(text) or str(text).strip() == "":
        return (1, "[物理截断]", "输入文本为空", 1.0) if mode == "Detailed" else (1, None, None, None)
    
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
            
            # 排雷点 2：严格匹配 Prompt 中定义的中文 Key
            label = res.get("label", 1) # 默认脏数据
            dirty_types = res.get("脏数据类型", [])
            analysis_reason = res.get("reason", "未提供理由")
            confidence = res.get("confidence", 0.0)
            
            if label == 0:
                dirty_str = ""
                reason_str = ""
            else:
                # 转换数组为字符串，方便 Excel 查看
                if isinstance(dirty_types, list):
                    dirty_str = ", ".join(str(r) for r in dirty_types)
                else:
                    dirty_str = str(dirty_types)
                reason_str = str(analysis_reason)
            
            return (label, dirty_str, reason_str, confidence) if mode == "Detailed" else (label, None, None, None)
                
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0)
                continue
            # 排雷点 3：错误状态下也返回 4 元组，防止 zip/concat 错位
            return ("Error", "API异常", f"异常详情: {str(e)}", 0.0)

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
        st.error("❌ 请检查 API Key 和文件是否已上传")
    else:
        try:
            ext = file_input.name.split('.')[-1].lower()
            df = pd.DataFrame()

            # 排雷点 4：增强编码识别和 CSV 解析稳定性
            if ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_input)
            elif ext == 'json':
                df = pd.read_json(file_input)
            elif ext in ['csv', 'txt']:
                raw_data = file_input.read()
                # 优先识别 BOM 编码
                if raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
                    enc = 'utf-16'
                elif raw_data.startswith(b'\xef\xbb\xbf'):
                    enc = 'utf-8-sig'
                else:
                    det = chardet.detect(raw_data)
                    enc = det['encoding'] if det['encoding'] else 'utf-8'
                
                try:
                    # 尝试自动分隔符识别，若失败则退回逗号分隔
                    df = pd.read_csv(io.BytesIO(raw_data), encoding=enc, sep=None, engine='python')
                except:
                    df = pd.read_csv(io.BytesIO(raw_data), encoding='gb18030')

            text_cols = [c for c in df.select_dtypes(include=['object']).columns]
            if not text_cols:
                st.error("❌ 无法在文件中找到文本列，请检查数据格式")
                st.stop()
                
            # 找到平均长度最长的列作为审计列
            target_col = df[text_cols].apply(lambda x: x.astype(str).str.len()).mean().idxmax()
            
            client = OpenAI(api_key=api_input, base_url=BASE_URL)
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 执行审计
            total_rows = len(df)
            for i, text in enumerate(df[target_col]):
                res = get_prediction(text, client, run_mode)
                results.append(res)
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"⏳ 正在审计第 {i+1}/{total_rows} 条数据...")
            
            # 排雷点 5：强制对齐校验，防止数据错位
            if len(results) != len(df):
                st.error(f"⚠️ 数据对齐异常！原始数据 {len(df)} 行，审计结果 {len(results)} 行。")
                st.stop()

            if run_mode == "Detailed":
                res_df = pd.DataFrame(results, columns=['Label', '脏数据类型', 'Reason', 'Confidence'])
            else:
                # Fast 模式只取 Label
                res_df = pd.DataFrame([r[0] for r in results], columns=['Label'])
            
            # 合并结果
            final_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
            
            # 生成下载文件
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False)
            
            st.success(f"✅ 审计完成！分析列：【{target_col}】")
            st.download_button("📥 下载审计报告", data=output.getvalue(), file_name=f"audit_report_{int(time.time())}.xlsx")
            
        except Exception as e:
            st.error(f"❌ 运行过程中出现错误: {str(e)}")
