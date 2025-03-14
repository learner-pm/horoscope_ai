import os
import pickle
import jieba

# 加载 Tokenizer
tokenizer_path = os.path.join(os.getcwd(), "models", "tokenizer.pkl")
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer 文件 {tokenizer_path} 不存在！")

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# ✅ 测试数据：包含星座和关键字
test_texts = ["白羊座的性格很有魅力", "金牛座的运势不错", "幸运色是红色", "天蝎座的事业充满挑战"]
test_tokens = [list(jieba.cut(text)) for text in test_texts]  # jieba 分词
test_sequences = tokenizer.texts_to_sequences([" ".join(tokens) for tokens in test_tokens])  # 转换为 ID

# ✅ 打印转换结果
print("=" * 60)
print("🚀  测试 Tokenizer 是否正确转换文本 🚀")
print("=" * 60)

for text, tokens, seq in zip(test_texts, test_tokens, test_sequences):
    print(f"原始文本: {text}")
    print(f"分词结果: {tokens}")
    print(f"转换为 ID: {seq}")
    print("-" * 50)

# ✅ 检查 OOV
oov_index = tokenizer.word_index.get("<OOV>", None)
if oov_index:
    print(f"<OOV> 词的 ID: {oov_index}")

for i, seq in enumerate(test_sequences):
    if not seq:
        print(f"⚠ 警告: '{test_texts[i]}' 无法转换为 ID，可能是 OOV！")
    elif oov_index in seq:
        print(f"⚠ 警告: '{test_texts[i]}' 转换后包含 OOV！")

print("=" * 60)
print("✅  测试完成！请检查转换结果！")
print("=" * 60)


# ✅ 检查 OOV 词是否在 tokenizer 里
check_words = ["白羊座", "性格", "魅力", "幸运色", "红色", "挑战"]
for word in check_words:
    print(f"'{word}' ID:", tokenizer.word_index.get(word, "❌ OOV"))


# ✅ 统计 tokenizer 词表大小
print("词表大小:", len(tokenizer.word_index))

# ✅ 查看最常见的 50 个词
top_words = sorted(tokenizer.word_index.items(), key=lambda x: x[1])[:50]
print("前 50 个单词:", top_words)