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

# ================= 2. 深度审计提示词 (V3.2 最新逻辑版) =================
SYSTEM_PROMPT = """你是一名极度专业且严谨的数据审计专家。你的任务是判定输入文本是否为“逻辑完备且可用于教学考核”的标准题目。

### 一、 分类核心哲学
- **标准 (0)**：有完整的考核任务（包括陈述式指令），逻辑链条闭合。允许背景中存在无关噪音（如水印、题号），只要不干扰题目主体的理解。
  - **【绝对排除项】**：任何包含答案、解析、过程推导、参考评分标准的文本，即便逻辑再完整，也必须判定为 **脏 (1)**。
- **脏 (1)**：逻辑链条断裂、关键参数缺失、包含答案解析、或是纯粹的非题目干扰。

### 二、 判定红线（出现以下情况必判为 1）

1. **逻辑空洞（对应：[物理截断]）**：
   - **已知量缺失**：文本中明确提到“已知”、“如图”、“如下表”，但后续没有具体的数值、描述或是数据。
   - **指令中途崩线**：在发出“请写出”、“计算...”、“求...”等指令后，紧接着出现非自然语言（如 [System Log]、[Error]、<br>、乱码、或突然切换到无关话题），导致原始考核任务无法读取。
   - **待求量缺失**：仅有背景陈述或公式罗列，没有任何提问或要求。
   - **语义截断**：文本在连词（因为、但是、如果）或公式中段突然中止，导致无法理解完整意图。

2. **纯粹废料与泄露（对应：[文本噪声]）**：
   - **结果严重泄露（高优先级）**：题干中直接包含了详细的解析步骤、标准答案、解题过程或结论。只要出现此类信息，一律视为破坏了考核纯度。
   - **非题目主体**：纯广告、语境孤儿（如“答案选A”）、无意义字符堆砌。

3. **格式污染（对应：[代码噪声]）**：
   - 包含机器语言残留（HTML/SQL/JSON等）或系统日志（NaN、ID、[音频]）、或是 LaTeX 渲染报错。

### 三、 豁免原则（以下情况必须判为 0）
1. **形式豁免**：以“简析”、“说明”等动词开头的陈述式指令，只要考核目标明确且不含答案，严禁判定为不完整。
2. **轻微噪音豁免**：主体逻辑闭合且无答案的情况下，粘连的水印、流水号等视为可接受干扰。

### 四、 强制审计流程（全维度扫描机制）
**注意：必须扫描完以下所有维度，严禁在发现第一个问题后停止审计。**
1. **答案解析初筛**：检索文本是否包含任何解题过程、推导步骤或标准答案。若有，记录 [文本噪声]。
2. **任务完整性扫描**：检查“已知量”与“待求量”是否齐备。若题目在公式中段中止，或选项出现缺失/乱码，导致无法完成考核，记录 [物理截断]。
3. **内容纯净度扫描**：检查是否有广告、水印或与题目无关的文字。若有，记录 [文本噪声]。
4. **格式合规性扫描**：检查是否有机器代码残留（<html>、sql等）、LaTeX 渲染错误。若有，记录 [代码噪声]。

### 五、 输出要求（严禁偏离）
请严格按照以下 JSON 模板输出，严禁擅自更改或增加 Key 名称：
```json
{
  "label": 1, 
  "脏数据类型": ["[文本噪声]", "[代码噪声]"], 
  "reason": "这里写详细的分析过程...",
  "confidence": 0.95
}"""

# ================= 3. 核心处理逻辑 =================

def get_prediction(text, client, mode):
    if pd.isna(text) or str(text).strip() == "":
        return (1, "[物理截断]", "文本为空", 1.0) if mode == "Detailed" else (1, None, None, None)
    
    for attempt in range(3):
        try:
            completion = client.chat.completion.create(
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
            # 严格对应提示词中的 JSON Key
            dirty_types = res.get("脏数据类型", [])
            analysis_text = res.get("reason", "") # 这里的 reason 存放分析文本
            confidence = res.get("confidence", 0.0)
            
            # 继承业务逻辑：标签为0时，清空类型和分析原因
            if label == 0:
                dirty_str = ""
                reason_str = ""
            else:
                dirty_str = ", ".join(str(r) for r in dirty_types) if isinstance(dirty_types, list) else str(dirty_types)
                reason_str = str(analysis_text)
            
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
                st.error("无法识别文本列，请检查文件格式")
                st.stop()
            
            # 自动寻找文本最长的一列
            target_col = df[text_cols].apply(lambda x: x.astype(str).str.len()).mean().idxmax()
            
            client = OpenAI(api_key=api_input, base_url=BASE_URL)
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, text in enumerate(df[target_col]):
                res = get_prediction(text, client, run_mode)
                results.append(res)
                progress_bar.progress((i + 1) / len(df))
                status_text.text(f"正在处理第 {i+1}/{len(df)} 条数据...")
            
            if run_mode == "Detailed":
                res_df = pd.DataFrame(results, columns=['Label', '脏数据类型', 'Reason', 'Confidence'])
            else:
                res_df = pd.DataFrame([r[0] for r in results], columns=['Label'])
            
            final_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False)
            
            st.success(f"✅ 审计完成！已分析：【{target_col}】列")
            st.download_button("📥 下载审计报告", data=output.getvalue(), file_name="audit_report.xlsx")
            
        except Exception as e:
            st.error(f"出错: {e}")
