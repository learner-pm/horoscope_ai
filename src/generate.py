import os
import pickle
import numpy as np
import jieba
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import config

def sample_with_temperature(preds, temperature=1.0):
    """
    根据预测的概率分布进行温度采样。
    preds：预期为1D数组（词汇表长度的概率分布）
    temperature：温度值，控制生成的多样性
    """
    # 确保 preds 是 1D 数组
    preds = np.asarray(preds).flatten().astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # 温度缩放
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)  # 归一化为概率分布
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class HoroscopeGenerator:
    def __init__(self, model_path, tokenizer):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在！")
        
        self.model = load_model(model_path)
        self.tokenizer = tokenizer
        self.max_length = config.max_length  # 使用配置文件中的最大长度

    def generate(self, seed_text, num_words=50, temperature=0.9):
        # 将种子文本分词并转换为数字序列
        seed_tokens = jieba.cut(seed_text)
        seed_sequence = self.tokenizer.texts_to_sequences([" ".join(seed_tokens)])[0]
        
        # 开始生成文本
        for i in range(num_words):
            # 填充序列，确保输入长度为 max_length
            padded_seq = pad_sequences([seed_sequence], maxlen=self.max_length, padding="post")
            
            # 预测结果 shape: (1, max_length, vocab_size)
            predictions = self.model.predict(padded_seq)
            # 选择最后一个时间步的预测结果，得到1D数组，长度为 vocab_size
            predicted_probs = predictions[0][-1]
            
            # 如果存在 <OOV> 标记，将其概率设置为0
            oov_index = self.tokenizer.word_index.get("<OOV>")
            if oov_index is not None and oov_index < len(predicted_probs):
                predicted_probs[oov_index] = 0
            
            # 温度采样生成下一个词的索引
            predicted_id = sample_with_temperature(predicted_probs, temperature)
            seed_sequence.append(predicted_id)
            
            # 如果生成的单词为句号、感叹号或问号，则停止生成
            generated_word = self.tokenizer.sequences_to_texts([[predicted_id]])[0]
            if generated_word in ['。', '！', '？']:
                break
        
        # 将生成的词序列转换为文本并返回
        generated_text = self.tokenizer.sequences_to_texts([seed_sequence])[0]
        return generated_text.replace(" ", "")  # 移除空格

if __name__ == "__main__":
    # 构造 tokenizer 的绝对路径
    tokenizer_path = os.path.join(os.getcwd(), "models", "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer 文件 {tokenizer_path} 不存在！")
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    # 构造模型文件的绝对路径
    model_path = os.path.join(os.getcwd(), "models", "lstm_horoscope.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")
    
    model = load_model(model_path)

    # 输出模型总结
    model.summary()
    seed_text = input("请输入星座名（默认白羊）：") or "白羊座"
    generator = HoroscopeGenerator(model_path, tokenizer)
    print("正在生成内容，请稍候...")
    print(generator.generate(seed_text))
