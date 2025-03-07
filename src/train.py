# -*- coding: utf-8 -*- 
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from model import build_lstm_model
from data_preprocess import load_and_clean_data, tokenize_texts
from config import config

def train_model():
    # 加载和预处理数据
    texts = load_and_clean_data()
    tokenizer, sequences = tokenize_texts(texts)
    
    # 划分输入和输出（自回归任务）
    X = sequences  # 输入：完整序列，形状为 (num_samples, 50)
    y = np.roll(sequences, -1, axis=1)  # 输出：序列向右移动一位，形状为 (num_samples, 50)
    
    # 将最后一个词设为填充标记（可选）
    y[:, -1] = 0  # 假设 0 是填充标记
    
    # 展平标签
    y = y.flatten()  # 形状从 (num_samples, 50) 变为 (num_samples * 50,)
    
    # 打印形状
    print("X 的形状:", X.shape)
    print("y 的形状:", y.shape)
    
    # 构建模型
    model = build_lstm_model(tokenizer)
    
    # 训练模型
    history = model.fit(
        X, y,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split
    )
    print("当前工作目录：", os.getcwd())
    # 保存模型
    model_path = "../models/lstm_horoscope.keras"
    print("开始保存模型...")
    model.save(model_path)
    print("保存模型完成！")
    print(f"模型已保存到 {model_path}")
    # 检查文件是否存在
    if os.path.exists(model_path):
        print(f"文件 {model_path} 已成功保存，大小为 {os.path.getsize(model_path)} 字节")
    else:
        print(f"文件 {model_path} 未保存成功")

if __name__ == "__main__":
    train_model()
