import os
# 设置Hugging Face相关的端点地址为镜像地址，目的可能是为了加速模型等资源的下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
# 从transformers库中导入自动分词器和自动模型类
from transformers import AutoTokenizer, AutoModel

# 设置可见的CUDA设备，这里设置为使用编号为0的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------------加载模型-----------------#


# 从预训练模型中加载分词器，模型名称为"THUDM/chatglm3-6b"，并信任远程代码（可能用于加载自定义或特殊的代码逻辑）
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
# 从预训练模型中加载模型，模型名称为"THUDM/chatglm3-6b"，信任远程代码，并将模型放置在CUDA（GPU）设备上
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
# 将模型设置为评估模式，关闭一些在训练过程中才会用到的操作（如梯度计算等）
model = model.eval()         
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-chat-7b-base", trust_remote_code=True)
# model = AutoModel.from_pretrained(
#     "deepseek-ai/deepseek-chat-7b-base",
#     trust_remote_code=True,
#     device_map="cuda"  # 自动处理模型在多GPU上的分布
# )
# model = model.eval()



# ---------------------------------#


def generate_patient_prompt(patient_data):
    """
    根据患者数据生成用于个性化描述的结构化提示语
    """
    # 从患者数据字典中获取大五人格得分，如果不存在则返回一个空字典
    big5_scores = patient_data.get("big5_scores", {})
    # 从患者数据字典中获取年龄，如果不存在则设置为"unknown"
    age = patient_data.get("age", "unknown")
    # 从患者数据字典中获取性别，如果不存在则设置为"unknown"
    gender = patient_data.get("gender", "unknown")
    # 从患者数据字典中获取籍贯，如果不存在则设置为"unknown"
    native_place = patient_data.get("native_place", "unknown")
    # 以下几行代码被注释掉，原本可能是想从患者数据中获取更多信息（如财务压力、家庭成员、疾病等）
    # financial_stress = patient_data.get("family_factors", {}).get("Financial_Stress", "unknown")
    # family_members = patient_data.get("family_factors", {}).get("Family_Members", "unknown")
    # disease = patient_data.get("disease", "unknown")

    # 从大五人格得分字典中分别获取各个维度的得分，如果不存在则设置为"unknown"
    extroversion = big5_scores.get("Extraversion", "unknown")
    agreeableness = big5_scores.get("Agreeableness", "unknown")
    openness = big5_scores.get("Openness", "unknown")
    neuroticism = big5_scores.get("Neuroticism", "unknown")
    conscientiousness = big5_scores.get("Conscientiousness", "unknown")

    # 以下代码被注释掉，原本可能是想对财务压力和疾病状态进行解释并映射为相应的描述字符串
    # Explain financial stress and disease
    # financial_stress_desc = {
    #     0: "no financial stress",
    #     1: "mild financial stress",
    #     2: "moderate financial stress",
    #     3: "severe financial stress"
    # }.get(financial_stress, "unknown financial stress level")

    # disease_desc = {
    #     "0": "the patient is healthy",
    #     "1": "the patient has other diseases",
    #     "2": "the patient has endocrine diseases",
    #     "3": "the patient has circulatory system diseases",
    #     "4": "the patient has neurological diseases"
    # }.get(disease, "unknown disease status")

    # 构建最终的提示语字符串，包含患者的基本信息、大五人格得分等，并提出了生成描述的要求
    prompt = (
        f"The patient is a {age}-year-old {gender} from {native_place}. "
        f"The patient's Extraversion score is {extroversion}. "
        f"The Agreeableness score is {agreeableness}. "
        f"The Openness score is {openness}. "
        f"The Neuroticism score is {neuroticism}. "
        f"The Conscientiousness score is {conscientiousness}. "
        # f"Their financial stress is categorized as {financial_stress_desc}, and they live with {family_members} family members. "
        # f"Based on the disease classification, {disease_desc}. "
        "Please generate a concise, fluent English description summarizing the patient's key personality traits, family environment, and other notable characteristics. "
        "Avoid mentioning depression or related terminology. "
        "Output the response as a single paragraph."
    )

    return prompt


def process_dataset(json_file, output_file):
    """
    处理JSON数据集并生成个性化描述
    """
    # 以只读模式打开输入的JSON文件，并将其内容解析为Python对象（通常是字典或列表）
    with open(json_file, "r") as f:
        dataset = json.load(f)

    # 初始化一个空字典，用于存储每个患者ID及其对应的个性化描述结果
    results = {}

    # 以写入模式打开输出文件
    with open(output_file, "w") as f:
        # 遍历数据集中的每个患者ID和对应的患者数据
        for patient_id, patient_data in dataset.items():
            print(f"Processing patient ID: {patient_id}")
            # 调用generate_patient_prompt函数为当前患者生成提示语
            patient_prompt = generate_patient_prompt(patient_data)
            print(f"Generated prompt for patient {patient_id}: {patient_prompt}")

            # 使用模型的chat方法，传入分词器、生成的提示语、空的对话历史和设定的温度参数（控制生成文本的随机性）来生成个性化回复
            response, history = model.chat(tokenizer, patient_prompt, history=[], temperature=0.1)
            print(f"Generated description for patient {patient_id}: {response}")

            # 将当前患者ID和生成的个性化描述存储到结果字典中
            results[patient_id] = response

        # 将结果字典以JSON格式写入输出文件，确保非ASCII字符正常显示，并设置缩进为4以增加可读性
        json.dump(results, f, ensure_ascii=False, indent=4)
        f.write("\n")  # 添加一个换行符，使输出的JSON文件更易读

    print(f"All patient descriptions saved to {output_file}.")
    

if __name__ == "__main__":
    # 输入的数据集JSON文件路径
    json_file = "/root/MPDD-Young/Training/labels/personalized_train.json"
    # 输出的包含个性化描述的JSON文件路径
    output_file = "/root/MPDD-Young/Training/labels/personalized.json"

    # 调用process_dataset函数处理数据集并生成个性化描述
    process_dataset(json_file, output_file)
