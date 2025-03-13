from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional  # 导入 Bidirectional
from config import config

def build_lstm_model(tokenizer):
    """构建LSTM模型，采用双向LSTM和序列到序列预测"""
    vocab_size = len(tokenizer.word_index) + 1  # 实际词汇表大小
    
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=config.embedding_dim,
            input_length=config.max_length  # 输入长度为 config.max_length
        ),
        Bidirectional(LSTM(config.lstm_units, return_sequences=True)),  # 双向 LSTM，返回序列
        Dropout(config.dropout_rate),
        LSTM(config.lstm_units, return_sequences=True),  # 修改这里，返回整个序列
        Dense(vocab_size, activation="softmax")  # 对序列中每个时间步进行分类
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # 使用 sparse_categorical_crossentropy
        metrics=["accuracy"]
    )
    return model
