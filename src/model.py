from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional  # 导入 Bidirectional
from config import config

def build_lstm_model(tokenizer):
    """构建LSTM模型"""
    vocab_size = len(tokenizer.word_index) + 1  # 实际词汇表大小
    
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=config.embedding_dim,
            input_length=config.max_length  # 输入长度为 50
        ),
        Bidirectional(LSTM(config.lstm_units, return_sequences=True)),  # 使用双向LSTM
        Dropout(config.dropout_rate),
        LSTM(config.lstm_units),
        Dense(vocab_size, activation="softmax")  # 输出形状为 (batch_size, vocab_size)
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # 使用 sparse_categorical_crossentropy
        metrics=["accuracy"]
    )
    return model
