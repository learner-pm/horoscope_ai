class Config:
    data_folder = "data/raw"  # 直接读取 data/raw 文件夹下所有 CSV
    max_length = 150  # 句子最大长度，统计数据后调整
    vocab_size = 40000  # 增加词汇表大小
    embedding_dim = 512  # 提高词向量维度
    lstm_units = 256  # 增加 LSTM 神经元
    dropout_rate = 0.3  # 防止过拟合

    batch_size = 64  # 增加批大小，提升训练速度
    epochs = 1  # 训练轮次增加
    validation_split = 0.1  # 验证集减少到 10%

config = Config()

