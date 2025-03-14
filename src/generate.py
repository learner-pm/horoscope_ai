import os
import pickle
import numpy as np
import jieba
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import config

def sample_top_k(preds, k=5):
    """
    Top-K 采样策略，防止模型总是选择概率最高的词
    preds: 预测的概率分布
    k: 选择前 k 个最高概率的词进行采样
    """
    preds = np.asarray(preds).astype("float64")
    top_k_indices = np.argsort(preds)[-k:]  # 取前 k 个概率最高的词
    top_k_probs = preds[top_k_indices]  # 获取对应概率
    top_k_probs /= np.sum(top_k_probs)  # 归一化概率
    
    return np.random.choice(top_k_indices, p=top_k_probs)  # 按 top-k 概率采样

class HoroscopeGenerator:
    def __init__(self, model_path, tokenizer):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在！")
        
        self.model = load_model(model_path)
        self.tokenizer = tokenizer
        self.max_length = config.max_length

        # ✅ 手动确保星座名在 `tokenizer`
        self.ensure_zodiac_tokens()

    def ensure_zodiac_tokens(self):
        """
        确保 `tokenizer` 包含所有星座名称，如果不存在则添加
        """
        zodiac_names = ["白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座", 
                        "天秤座", "天蝎座", "射手座", "摩羯座", "水瓶座", "双鱼座"]
        
        for zodiac in zodiac_names:
            if zodiac not in self.tokenizer.word_index:
                self.tokenizer.word_index[zodiac] = len(self.tokenizer.word_index) + 1
                print(f"⚠ 添加 OOV 词汇 {zodiac} 到 tokenizer")

    def generate(self, seed_text, num_words=50, k=5):
        """
        生成星座文本，采用 Top-K 采样
        """
        if not seed_text.strip():
            raise ValueError("输入不能为空！")

        seed_tokens = list(jieba.cut(seed_text))
        print("分词结果:", seed_tokens)

        seed_sequence = self.tokenizer.texts_to_sequences([" ".join(seed_tokens)])[0]
        if not seed_sequence:
            print("⚠ 种子文本未能转换为词 ID，可能是 OOV 词！")
            return "<OOV>"

        generated_sequence = seed_sequence[:]

        for _ in range(num_words):
            padded_seq = pad_sequences([generated_sequence[-self.max_length:]], maxlen=self.max_length, padding="pre")
            predictions = self.model.predict(padded_seq, verbose=0)
            predicted_probs = predictions[0][-1]  # 取最后一个时间步的预测分布

            # Top-K 采样
            predicted_id = sample_top_k(predicted_probs, k)

            generated_sequence.append(predicted_id)
            generated_word = self.tokenizer.sequences_to_texts([[predicted_id]])[0]

            # 遇到句子终止符，提前结束
            if generated_word in ["。", "！", "？"]:
                break

        generated_text = self.tokenizer.sequences_to_texts([generated_sequence])[0]
        return generated_text.replace(" ", "")


if __name__ == "__main__":
    tokenizer_path = os.path.join("models", "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer 文件 {tokenizer_path} 不存在！")
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    model_path = os.path.join("models", "lstm_horoscope.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")

    generator = HoroscopeGenerator(model_path, tokenizer)

    zodiac_names = ["白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座", 
                    "天秤座", "天蝎座", "射手座", "摩羯座", "水瓶座", "双鱼座"]

    for zodiac in zodiac_names:
        print(f"正在为 {zodiac} 生成内容...")
        generated_text = generator.generate(zodiac, k=5)
        print(f"生成内容：{generated_text}\n")
