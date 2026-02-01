import torch
import json
from transformers import BertTokenizerFast,BertTokenizer, BertForTokenClassification

# ========== 加载模型和标签映射 ==========
MODEL_PATH = "./outputs/slot_filling/checkpoint-94"  # 训练好的槽位抽取模型输出目录

# 加载slot标签与id映射，用于id和标签之间转换
with open("data/slot_label2id.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# 加载分词器和训练好的BERT+Token分类模型
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()  # 设置为评估模式，禁用dropout等

def predict_slots(tokens):
    """
    槽位抽取推理入口。
    输入: tokens (分词后的token列表，如["打开", "空调", "到", "22", "度"])
    输出: token及其BIO槽位标签组成的list
    """
    # 对输入tokens进行编码，is_split_into_words=True表示输入已分词
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=32)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = logits.argmax(-1)[0].tolist()
    word_ids = inputs.word_ids(batch_index=0)

    labels, previous_word_idx = [], None
    for pred_id, word_idx in zip(pred_ids, word_ids):
        if word_idx is None:           # 跳过特殊 token
            continue
        if word_idx != previous_word_idx:
            labels.append(id2label.get(pred_id, "O"))
            previous_word_idx = word_idx
        # 如果希望把同一词的后续子词也映射成 I- 标签，可在这里加逻辑

    return list(zip(tokens, labels))

if __name__ == "__main__":
    # 示例输入：可替换为任何分词token列表
    tokens = ["打开", "空调", "调到", "20", "度"]
    #tokens = ["打开", "空调", "调到", "22", "度"]
    print("槽位抽取结果:", predict_slots(tokens))   # 输出形如[("打开", "O"), ("空调", "B-facility"), ...]