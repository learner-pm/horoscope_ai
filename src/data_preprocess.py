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
    """加载并清洗原始数据"""
    print("当前工作目录:", os.getcwd())
    print("数据文件路径:", os.path.abspath(config.data_path))
    
    try:
        df = pd.read_csv(config.data_path, encoding="utf-8")
    except FileNotFoundError:
        print(f"错误：文件 {config.data_path} 不存在！")
        return []
    
    texts = df["text"].tolist()
    
    # 清洗文本
    cleaned_texts = []
    for text in texts:
        text = re.sub(r"[^\u4e00-\u9fa5，。！？]", "", text)  # 去除非中文字符
        cleaned_texts.append(text.strip())
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