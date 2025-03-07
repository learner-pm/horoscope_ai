class Config:
    data_path = "data/raw/horoscope_data.csv"

    max_length = 100  # 每条文本的最大长度
    vocab_size = 20000  # 增大词汇表大小
    embedding_dim = 256  # 增大词向量维度
    lstm_units = 256  # LSTM神经元数量
    dropout_rate = 0.2  # Dropout比例

    batch_size = 64  # 批大小
    epochs = 10  # 增加训练轮数
    validation_split = 0.2  # 验证集比例

config = Config()

