import pickle
import pandas as pd
import re
import os
import jieba
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import config

zodiac_names = ["白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座", "天秤座", "天蝎座", "射手座", "摩羯座", "水瓶座", "双鱼座"]
for zodiac in zodiac_names:
    jieba.add_word(zodiac)

for zodiac in zodiac_names:
    print(f"{zodiac} 是否被正确分词:", list(jieba.cut(zodiac)))
    
def load_and_clean_data():
    """加载并清洗 data/raw 文件夹下的所有 CSV 文件"""
    data_folder = config.data_folder  # 直接读取 data/raw 文件夹
    if not os.path.exists(data_folder):
        print(f"错误：数据文件夹 {data_folder} 不存在！")
        return []

    all_texts = []
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):  # 只处理 CSV 文件
            file_path = os.path.join(data_folder, file)
            print(f"正在加载文件: {file_path}")

            try:
                # ✅ 读取CSV文件，新的格式只有一列
                df = pd.read_csv(file_path, encoding="utf-8", header=0, names=["text"])

                # ✅ 取 text 列，不取 sign
                texts = df["text"].dropna().tolist()
                all_texts.extend(texts)

            except Exception as e:
                print(f"加载 {file} 时出错: {e}")

    # ✅ 清洗文本：保留中文字符及常用标点（，。！？）
    cleaned_texts = [re.sub(r"[^\u4e00-\u9fa5，。！？]", "", text).strip() for text in all_texts]

    print(f"✅ 成功加载 {len(cleaned_texts)} 条数据")
    return cleaned_texts


def tokenize_texts(texts):
    """分词并构建词汇表"""
    # ✅ 分词（确保 `zodiac_names` 也被分词）
    tokenized_texts = [" ".join(jieba.cut(text)) for text in texts]

    # ✅ 训练 tokenizer
    tokenizer = Tokenizer(num_words=config.vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(tokenized_texts)

    # ✅ 转换为数字序列
    sequences = tokenizer.texts_to_sequences(tokenized_texts)
    padded_sequences = pad_sequences(sequences, maxlen=config.max_length, padding="post", truncating="pre")

    # ✅ 保存 tokenizer 到 models 目录
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    tokenizer_path = os.path.join(models_dir, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✅ tokenizer.pkl 已保存到 {tokenizer_path}")
    seq_lengths = [len(seq) for seq in sequences]
    print(f"最大 token 长度: {max(seq_lengths)}")
    print(f"最小 token 长度: {min(seq_lengths)}")
    print(f"平均 token 长度: {sum(seq_lengths) / len(seq_lengths):.2f}")
    return tokenizer, padded_sequences


if __name__ == "__main__":
    texts = load_and_clean_data()
    if texts:
        tokenizer, sequences = tokenize_texts(texts)
        print("前 50 个词:", list(tokenizer.word_index.items())[:50])

        # 保存处理后的数据（可选）
        np.save("data/processed/sequences.npy", sequences)
        print("✅ 数据预处理完成！")
