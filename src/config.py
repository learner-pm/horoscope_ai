class Config:
    data_folder = "data/raws"  # 直接读取 data/raw 文件夹下所有 CSV
    max_length = 150  # 句子最大长度，保持不变
    vocab_size = 10000  # 词汇表大小：根据数据量增大，5w+ 数据可以设置为 10000 左右
    embedding_dim = 512  # 提高词向量维度，512 通常是较为合适的选择
    lstm_units = 512  # 增加 LSTM 神经元，50000+ 数据下可以考虑 512 或更高
    dropout_rate = 0.3  # 防止过拟合，维持为 0.3 合适

    batch_size = 512  # 增加批大小，提升训练速度
    epochs = 4  # 训练轮次，保持在 4-5 轮之间，适合较大的数据集
    validation_split = 0.1  # 验证集比例，10% 的验证集

config = Config()
