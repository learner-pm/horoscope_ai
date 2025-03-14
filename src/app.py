from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import jieba
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from config import config

app = Flask(__name__)

# 加载 tokenizer 和模型
tokenizer_path = "models/tokenizer.pkl"
model_path = "models/lstm_horoscope.keras"

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

model = load_model(model_path)

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_text, num_words=50, temperature=0.9):
    # 分词和转换为数字序列
    seed_tokens = jieba.cut(seed_text)
    seed_sequence = tokenizer.texts_to_sequences([" ".join(seed_tokens)])[0]
    
    # 生成文本
    for i in range(num_words):
        padded_seq = pad_sequences([seed_sequence], maxlen=config.max_length, padding="post")
        predicted_probs = model.predict(padded_seq)[0][-1]
        
        # 避免 <OOV> 的影响
        oov_index = tokenizer.word_index.get("<OOV>")
        if oov_index is not None and oov_index < len(predicted_probs):
            predicted_probs[oov_index] = 0
        
        predicted_id = sample_with_temperature(predicted_probs, temperature)
        seed_sequence.append(predicted_id)
        
        # 检查是否生成结束标点
        generated_word = tokenizer.sequences_to_texts([[predicted_id]])[0]
        if generated_word in ['。', '！', '？']:
            break
    
    generated_text = tokenizer.sequences_to_texts([seed_sequence])[0]
    return generated_text.replace(" ", "")

# 定义 API 路由
@app.route('/generate', methods=['POST'])
def generate():
    # 获取用户请求中的文本和其他参数
    data = request.get_json()
    seed_text = data.get('seed_text', '白羊座')
    num_words = data.get('num_words', 50)
    temperature = data.get('temperature', 0.9)
    
    # 调用文本生成函数
    generated_text = generate_text(seed_text, num_words, temperature)
    
    return jsonify({'generated_text': generated_text})

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)