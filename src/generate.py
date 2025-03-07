import os
import pickle
import numpy as np
import jieba
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import config

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class HoroscopeGenerator:
    def __init__(self, model_path, tokenizer):
        # 确保路径正确
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在！")
        
        self.model = load_model(model_path)
        self.tokenizer = tokenizer
        self.max_length = config.max_length

    def generate(self, seed_text, num_words=50, temperature=0.9):
        seed_tokens = jieba.cut(seed_text)
        seed_sequence = self.tokenizer.texts_to_sequences([" ".join(seed_tokens)])[0]
        
        for i in range(num_words):
            padded_seq = pad_sequences([seed_sequence], maxlen=self.max_length, padding="post")
            predicted_probs = self.model.predict(padded_seq)[0]
            
            # 避免选择 <OOV> 标记（若存在）
            oov_index = self.tokenizer.word_index.get("<OOV>")
            if oov_index is not None and oov_index < len(predicted_probs):
                predicted_probs[oov_index] = 0
            
            # 使用温度采样
            predicted_id = sample_with_temperature(predicted_probs, temperature)
            seed_sequence.append(predicted_id)
        
        generated_text = self.tokenizer.sequences_to_texts([seed_sequence])[0]
        return generated_text.replace(" ", "") 

if __name__ == "__main__":
    # 加载 tokenizer
    tokenizer_path = "../models/tokenizer.pkl"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer 文件 {tokenizer_path} 不存在！")
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    # 确保模型路径正确
    model_path = "../models/lstm_horoscope.keras"  # 修改为实际的模型路径
    
    # 用户输入生成星座文本
    seed_text = input("请输入星座名（默认白羊）：") or "白羊座"  # 默认输入为白羊座
    generator = HoroscopeGenerator(model_path, tokenizer)
    print("正在生成内容，请稍候...")
    print(generator.generate(seed_text))
