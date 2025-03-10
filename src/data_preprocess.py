import pickle  # 添加这行
import pandas as pd
import re
import os
import jieba
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import config

def load_and_clean_data():
    """加载并清洗 data/raw 文件夹下的所有 CSV 文件"""
    data_folder = config.data_folder # 直接读取 data/raw 文件夹
    if not os.path.exists(data_folder):
        print(f"错误：数据文件夹 {data_folder} 不存在！")
        return []

    all_texts = []
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):  # 只处理 CSV 文件
            file_path = os.path.join(data_folder, file)
            print(f"正在加载文件: {file_path}")

            try:
                df = pd.read_csv(file_path, encoding="utf-8")
                if "text" not in df.columns:
                    print(f"警告：文件 {file} 缺少 'text' 列，跳过！")
                    continue
                
                texts = df["text"].dropna().tolist()  # 去除 NaN 值
                all_texts.extend(texts)

            except Exception as e:
                print(f"加载 {file} 时出错: {e}")
    
    # 清洗文本
    cleaned_texts = [re.sub(r"[^\u4e00-\u9fa5，。！？]", "", text).strip() for text in all_texts]
    
    print(f"成功加载 {len(cleaned_texts)} 条数据")
    return cleaned_texts


def tokenize_texts(texts):
    """分词并构建词汇表"""
    # 中文分词
    tokenized_texts = [" ".join(jieba.cut(text)) for text in texts]
    
    # 创建词汇表
    tokenizer = Tokenizer(num_words=config.vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(tokenized_texts)
    
    # 转换为数字序列
    sequences = tokenizer.texts_to_sequences(tokenized_texts)
    padded_sequences = pad_sequences(sequences, maxlen=config.max_length, padding="post")
    
    # 保存 tokenizer
    with open("../models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    
    return tokenizer, padded_sequences

if __name__ == "__main__":
    texts = load_and_clean_data()
    if texts:
        tokenizer, sequences = tokenize_texts(texts)
        
        # 保存处理后的数据（可选）
        np.save("data/processed/sequences.npy", sequences)
        print("数据预处理完成！")