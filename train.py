import json
import os
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

#基础参数设置
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCH = 100
LEARNING_RATE = 3e-5
OUTPUT_DIR = "./output"
CHECKPOINT_DIR = "./outputs/checkpointer-86"

#载入意图标签映射关系
with open("./data/intent_label_map.json",'r',encoding='utf-8') as f:
    label2id = json.load(f)
id2label = {v:k for k,v in label2id.items()}


#文本预处理、分词和label编码
def preprocess_function(examples):
    examples["label"] = [label2id.get(label) for label in examples["label"]]
    return tokenizer(
        examples["text"],
        truncation = True,
        padding="max_length",
        max_length = MAX_LENGTH,
    )

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)\
#数据集加载
raw_dateset = load_dataset(
    "json",
    data_files = {
        "train":"data/intent_train.json",
        "validation":"data/intent_dev.json"
    }
)

#应用分词并保存字段
encoded_datasets = raw_dateset.map(preprocess_function,batched=True)
columns = ["input_ids","attention_mask","label"]
encoded_datasets.set_format(type="torch", columns=columns)

#加载模型，如果有已保存的checkpoint 则从checkpoint加载
resume_from_checkpoint = None
if os.path.exists(CHECKPOINT_DIR) and os.path.isdir(CHECKPOINT_DIR):
    print(f"正在从checkpoint{CHECKPOINT_DIR}继续训练")
    model = BertForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    resume_from_checkpoint = CHECKPOINT_DIR
else:
    print("未发现 checkpoint ,将从预训练模型开始训练")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    resume_from_checkpoint = None

    #训练参数定义
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2
)
#验证集评测函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    accuracy = (preds == labels).sum() / len(labels)
    return {"accuracy": accuracy.item()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if __name__ == '__main__':
    #训练：可自动判断断点续练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    #保存最终模型和分词器
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)