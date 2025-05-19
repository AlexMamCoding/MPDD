import os
# 设置Hugging Face的端点为镜像地址，以加快模型下载速度
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import json
import numpy as np
import torch
# 从transformers库导入Roberta的分词器和模型
from transformers import RobertaTokenizer, RobertaModel


def load_data(json_file):
    # 打开指定的JSON文件并以只读模式读取
    with open(json_file, "r") as f:
        # 使用json.load函数将文件内容解析为Python对象
        data = json.load(f)
    return data


def generate_embeddings(descriptions, model, tokenizer, output_file):
    """
    Generate embeddings for each description and save them along with their IDs.
    """
    # 用于存储每个描述的ID和对应的嵌入向量
    embeddings_with_ids = []

    # 将模型设置为评估模式，关闭一些在训练时使用的特殊层（如Dropout）
    model.eval()  

    # 禁用梯度计算，以减少内存消耗并提高推理速度
    with torch.no_grad():  
        # 遍历描述字典中的每个ID和对应的描述
        for id_, description in descriptions.items():
            print(f"Processing ID: {id_}")
            # 使用分词器对描述进行分词，并将其转换为PyTorch张量
            # padding=True表示对输入进行填充，使它们具有相同的长度
            # truncation=True表示如果输入超过最大长度则进行截断
            # max_length=512表示最大长度为512
            encoded_input = tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # 将编码后的输入传递给模型，得到模型的输出
            output = model(**encoded_input)

            # 提取CLS标记的表示，通常用于表示整个句子的语义
            embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
            print(embedding.shape)

            # 将ID和对应的嵌入向量作为一个字典项添加到列表中
            embeddings_with_ids.append({"id": id_, "embedding": embedding})

    # 将包含ID和嵌入向量的列表保存为numpy数组
    np.save(output_file, embeddings_with_ids, allow_pickle=True)
    print(f"Embeddings and IDs saved to {output_file}")


def main():
    # 输入的JSON文件的路径
    json_file = "/root/MPDD-Young/Training/personalized.json"

    # 保存输出嵌入向量的文件路径
    output_file = "/root/MPDD-Young/Training/individualEmbedding/descriptions_embeddings_with_ids.npy"
    # 要加载的Roberta模型的名称
    model_name = "roberta-large"
    # 从预训练模型中加载Roberta的分词器
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # 从预训练模型中加载Roberta模型
    model = RobertaModel.from_pretrained(model_name)

    # 加载个性化描述数据
    descriptions = load_data(json_file)
    print(f"Loaded {len(descriptions)} descriptions.")

    # 生成并保存包含ID的嵌入向量
    generate_embeddings(descriptions, model, tokenizer, output_file)


if __name__ == "__main__":
    # 当脚本作为主程序运行时，调用main函数
    main()