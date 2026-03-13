import json
import os
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments

# ========== 基本配置参数 ==========
MODEL_NAME = "bert-base-chinese"        # BERT中文预训练模型名
MAX_LENGTH = 32                         # 输入序列最大长度
BATCH_SIZE = 16                         # 批次大小
EPOCHS = 100                             # 训练轮数
LEARNING_RATE = 3e-5                    # 学习率
OUTPUT_DIR = "./outputs/slot_filling/"  # 模型输出路径
CHECKPOINT_DIR = "./outputs/slot_filling/checkpoint-92" # 继续训练的断点保存目录
MODEL_PATH ="./output/checkpoint-200" #你自己的model的path

# ========== 加载标签映射 ==========
with open("data/slot_label2id.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(label2id)

# ========== 加载分词器（使用Fast版本支持 word_ids） ==========
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# ========== 读取数据集 ==========
raw_datasets = load_dataset(
    "json",
    data_files={"train": "data/slot_train.json", "validation": "data/slot_dev.json"}
)

# ========== 数据预处理与标签对齐函数 ==========
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                # B-标签后面遇到分词，标记为I-同槽位，否则用原label
                if label[word_idx].startswith("B-"):
                    label_ids.append(label2id[label[word_idx]] + 1)
                else:
                    label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# ========== 批量映射数据集 ==========
encoded_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)
columns = ["input_ids", "attention_mask", "labels"]
encoded_datasets.set_format(type="torch", columns=columns)

# ========== 加载模型（自动 checkpoint 恢复或初始化） ==========
resume_from_checkpoint = None    # 是否断点续练
if os.path.exists(CHECKPOINT_DIR) and os.path.isdir(CHECKPOINT_DIR):
    print(f"检测到 checkpoint，正在从 {CHECKPOINT_DIR} 恢复模型继续训练 ...")
    model = BertForTokenClassification.from_pretrained(CHECKPOINT_DIR)
    resume_from_checkpoint = CHECKPOINT_DIR
else:
    print("未检测到checkpoint，将从预训练模型全新训练 ...")
    model = BertForTokenClassification.from_pretrained(
        MODEL_PATH, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id,ignore_mismatched_sizes=True
    )
    resume_from_checkpoint = None

# ========== 设置训练参数 ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    save_strategy="epoch",                  # 每个epoch保存一次
    evaluation_strategy="epoch",            # 每个epoch评估
    logging_steps=20,
    logging_dir=f"{OUTPUT_DIR}/logs",
    save_total_limit=2                      # 仅保存最新2个checkpoint
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    print(preds)
    print(p.label_ids)
    labels = p.label_ids
    correct, total = 0, 0
    for p_, y_ in zip(preds, labels):
        for pi, yi in zip(p_, y_):
            if yi != -100:
                correct += int(pi == yi)
                total += 1
    return {"accuracy": correct / total if total > 0 else 0}

# ========== 创建Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ========== 主流程入口 ==========
if __name__ == "__main__":
    # 启动（可自动继续训练或全新训练）
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)