import os
from openai import OpenAI
import json
import time

# 初始化 DeepSeek 客户端

api_key = os.getenv("DEEPSEEK_API_KEY", "sk-9013c276a9234d6c93c6cca41661f913")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

# 定义你的目标场景和数量
SCENARIOS = {
    "SPORTS": "体育比赛比分，读作'比'",
    "TEMP_MATH": "温度、负数、股票跌幅，读作'负'",
    "RANGE": "时间、年份、距离的范围，读作'至'或'到'",
    "PHONE": "座机电话号码、各类编号，通常不发音或读作'杠'",
    "MATH_SUB": "数学减法运算，读作'减'",
    "CHEMISTRY": "化学同位素或特定专有名词，读作'杠' (如碳-14)"
}

OUTPUT_FILE = "train_data1227.jsonl"

def call_llm_api(prompt):
    """调用 DeepSeek API 生成内容"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的数据生成助手。请只返回纯 JSON 格式的数据，不要包含其他解释性文字。"},
                {"role": "user", "content": prompt}
            ],
            temperature=1.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用出错: {e}")
        raise

def parse_json(response_text):
    """解析 LLM 返回的 JSON 字符串"""
    # 尝试清理 Markdown 代码块标记
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("完整 JSON 解析失败，尝试逐条提取有效数据...")
        
        valid_objects = []
        decoder = json.JSONDecoder()
        
        # 尝试跳过开头的 [
        search_text = text
        if search_text.startswith('['):
            search_text = search_text[1:]
            
        pos = 0
        while True:
            search_text = search_text.strip()
            if not search_text:
                break
                
            try:
                # 尝试解析一个 JSON 对象
                obj, idx = decoder.raw_decode(search_text)
                valid_objects.append(obj)
                
                # 移动指针
                search_text = search_text[idx:]
                
                # 跳过逗号
                search_text = search_text.strip()
                if search_text.startswith(','):
                    search_text = search_text[1:]
                    
            except json.JSONDecodeError:
                # 如果解析失败（比如遇到截断的结尾），尝试寻找下一个可能的对象起始点
                # 或者直接停止
                next_start = search_text.find('{')
                if next_start != -1 and next_start > 0:
                     search_text = search_text[next_start:]
                else:
                    break
        
        if valid_objects:
            print(f"成功挽回 {len(valid_objects)} 条数据")
            return valid_objects
            
        print(f"无法提取有效数据。")
        return []

def generate_data(scenario_key, scenario_desc):
    prompt = f"""
    你是一个中文语料生成专家。请根据以下场景生成 50 条训练数据。
    场景描述：【{scenario_desc}】

    要求：
    1. 生成的句子必须包含连字符 '-'。
    2. 必须明确 '-' 在该语境下的正确读音（汉字）。
    3. 返回格式必须是纯 JSON 列表，不要包含 Markdown 标记。
    4. 列表中的每个元素是一个字典，包含两个字段：
       - "input": 包含连字符的中文句子。
       - "output": 连字符在该句子中的正确读音（仅限一个汉字或词，如"比"、"负"、"至"、"减"、"杠"）。

    参考示例：
    [
        {{"input": "最终比分定格在 105-98，主队获胜。", "output": "比"}},
        {{"input": "昨晚最低气温达到了 -15℃。", "output": "负"}},
        {{"input": "请参考第 10-15 页的内容。", "output": "至"}},
        {{"input": "咨询电话：021-88888888。", "output": "杠"}},
        {{"input": "5 - 3 = 2", "output": "减"}}
    ]

    请基于场景【{scenario_desc}】生成 50 条数据：
    """
    
    # 这里调用你的大模型 API (GPT-4, Qwen-Max 等)
    response = call_llm_api(prompt) 
    data_list = parse_json(response)
    
    # 转换为目标格式
    final_data = []
    for item in data_list:
        # 兼容可能返回的字段名，确保健壮性
        input_text = item.get("input", item.get("text", ""))
        output_text = item.get("output", item.get("pronunciation", ""))
        
        if input_text and output_text:
            final_data.append({
                "instruction": "请判断下列句子中连字符'-'的正确读音，直接输出读音汉字。",
                "input": input_text,
                "output": output_text
            })
    return final_data

# 主循环
total_count = 0
# 每次循环次数
BATCH_LOOPS = 50

for key, desc in SCENARIOS.items():
    print(f"\n=== 正在生成场景: {key} ===")

    for i in range(BATCH_LOOPS):
        print(f"[{key}] 进度: {i+1}/{BATCH_LOOPS} ... ", end="", flush=True)
        try:
            batch_data = generate_data(key, desc)
            
            if batch_data:
                # 实时追加写入文件
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    for item in batch_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                count = len(batch_data)
                total_count += count
                print(f"成功生成 {count} 条数据 (总计: {total_count})")
            else:
                print("未生成有效数据")
                
            time.sleep(1) # 防止 API Rate Limit
        except Exception as e:
            print(f"\nError: {e}")

print(f"\n全部生成完成，共 {total_count} 条数据。")