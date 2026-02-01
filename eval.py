import json
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

# ========== 参数定义（与训练脚本保持一致） ==========
MODEL_NAME = "bert-base-chinese"     # 使用的预训练BERT模型名
MAX_LENGTH = 64                      # 每条文本最大长度
BATCH_SIZE = 16                      # 批量大小
OUTPUT_DIR = "./outputs/checkpoint-134"            # 输出目录，含模型文件

# ========== 加载标签映射 ==========
# label2id: 标签字符串 -> 标签id，id2label: 标签id -> 标签字符串
with open("intent_label_map.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# ========== 加载分词器和模型 ==========
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# 加载训练好的意图分类模型
model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)
model.eval()

# ========== 定义文本预处理函数 ==========
def preprocess_function(examples):
    # 标签转id
    examples["label"] = [label2id.get(label) for label in examples["label"]]
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# ========== 加载和预处理验证集/测试集 ==========

eval_datasets = load_dataset(
    "json",
    data_files={"eval": "./data/intent_dev.json"}
)
encoded_eval_dataset = eval_datasets.map(preprocess_function, batched=True)
columns = ["input_ids", "attention_mask", "label"]
encoded_eval_dataset.set_format(type="torch", columns=columns)

# ========== 验证或测试 ==========
# 用Trainer封装便于与训练一致
trainer = Trainer(
    model=model,
    tokenizer=tokenizer
)

# 获取预测结果、真实标签
outputs = trainer.predict(encoded_eval_dataset["eval"])
preds = outputs.predictions.argmax(-1)
labels = outputs.label_ids

# 计算准确率
accuracy = (preds == labels).sum() / len(labels)
print(f"Eval Accuracy: {accuracy:.4f}")

# ========== 可选：输出详细预测结果 ==========
# 将每条预测的原文本、预测标签和真实标签打印/保存
with open("data/intent_dev.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

for i, data in enumerate(eval_data):
    pred_label = id2label[int(preds[i])]
    true_label = id2label[int(labels[i])]
    sent = data["text"]
    print(f"文本: {sent}\n真实意图: {true_label}\t预测意图: {pred_label}\n{'-'*40}")