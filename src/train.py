# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_lstm_model
from data_preprocess import load_and_clean_data, tokenize_texts
from config import config

def train_model():
    # ✅ 1. 加载数据
    texts = load_and_clean_data()
    tokenizer, sequences = tokenize_texts(texts)

    # ✅ 2. 处理 X, y
    X = sequences  # 输入序列，形状 (num_samples, max_length)
    y = np.roll(X, -1, axis=1)  # 右移一位
    y[:, -1] = 0  # 让最后一个 token 变为填充标记

    
    print(f"X 的形状: {X.shape}, y 的形状: {y.shape}")

    # ✅ 3. 打印形状（保留原始日志）
    print("成功加载", len(texts), "条数据")
    print("X 的形状:", X.shape)  # (num_samples, max_length-1)
    print("y 的形状:", y.shape)  # (num_samples, max_length-1)

    # ✅ 4. 手动拆分训练集和验证集（替代 validation_split）
    total_size = len(X)
    val_size = int(total_size * config.validation_split)
    train_size = total_size - val_size

    train_X, val_X = X[:train_size], X[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]

    # ✅ 5. 构建 tf.data.Dataset
    train_dataset = Dataset.from_tensor_slices((train_X, train_y)).shuffle(5000).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = Dataset.from_tensor_slices((val_X, val_y)).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    # ✅ 6. 构建 LSTM 模型
    model = build_lstm_model(tokenizer)

    # ✅ 7. 断点训练（防止训练中断时丢失进度）
    checkpoint_path = os.path.join("models", "lstm_checkpoint.keras")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    # ✅ 8. 开始训练（修正 validation_split 错误）
    print("当前工作目录：", os.getcwd())
    print("开始训练模型...")
    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=val_dataset,  # 直接传递验证集
        #callbacks=[checkpoint]  # 重新启用 checkpoint
    )

    # ✅ 9. 保存最终模型（确保路径正确）
    model_path = os.path.join("models", "lstm_horoscope.keras")
    print("开始保存模型...")
    model.save(model_path)
    print("保存模型完成！")
    print(f"模型已保存到 {model_path}")

    # ✅ 10. 检查文件是否成功保存
    if os.path.exists(model_path):
        print(f"文件 {model_path} 已成功保存，大小为 {os.path.getsize(model_path)} 字节")
    else:
        print(f"文件 {model_path} 未保存成功")

if __name__ == "__main__":
    train_model()
