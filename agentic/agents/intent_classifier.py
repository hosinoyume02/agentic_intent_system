import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, BertForTokenClassification
import torch.nn.functional as F

class IntentClassifier:
    def __init__(self, model_path):
        # 加载意图分类用的分词器和模型
        intent_path = f"{model_path}checkpoint-134"
        self.tokenizer = BertTokenizerFast.from_pretrained(intent_path)
        self.model = BertForSequenceClassification.from_pretrained(intent_path)
        self.model.eval()  # 设置评估模式，关闭dropout等
        self.id2label = self.model.config.id2label  # 取意图标签映射

        # 加载槽位抽取的模型和标签字典
        # 假设slot_filling模型和label2id文件在model_path/slot_filling下
        slot_path = f"{model_path}slot_filling/checkpoint-94"
        self.slot_model = BertForTokenClassification.from_pretrained(slot_path)
        self.slot_model.eval()  # 也设置为评估模式
        self.slot_tokenizer = BertTokenizerFast.from_pretrained(slot_path)

        # 槽位标签映射文件json加载
        with open(f"data/slot_label2id.json", "r", encoding="utf-8") as f:
            import json
            self.slot_label2id = json.load(f)
        # id到标签名的映射
        self.slot_id2label = {int(v): k for k, v in self.slot_label2id.items()}

    def predict(self, text):
        """
        对输入句子text，完成意图识别和槽位抽取。
        :param text: 原始用户输入字符串
        :return: dict，{"intent":..., "slots":{...}}
        """
        # 1. 文本分词和batch维处理
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=32)
        
        # 2. 意图识别（句子级分类），输出类别ID进而映射至标签
        with torch.no_grad():
            logits = self.model(**inputs).logits  #[0.1,0.7,0.2,0.1....]
            probs = F.softmax(logits, dim=-1)[0]   #[0.1,0.7,0.2,0.1....]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()
           
            # 分两类：字典和列表，兼容不同transformers版本
            intent = self.id2label[int(pred)] if isinstance(self.id2label, dict) else self.id2label[pred]
        
        # 3. 槽位抽取（序列标注），对每个token判定槽位类型
        slot_tags = []  # （token, BIO槽位标签）
        slot_inputs = self.slot_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=32,
            return_offsets_mapping=True,
        )
        offsets = slot_inputs.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            slot_logits = self.slot_model(**slot_inputs).logits
        slot_pred_ids = slot_logits.argmax(-1)[0].tolist()

        for pred_id, (start, end) in zip(slot_pred_ids, offsets):
            if start == end:
                continue  # 跳过 [CLS]/[SEP]/[PAD]
            piece = text[start:end]
            slot_label = self.slot_id2label.get(pred_id, "O")
            slot_tags.append((piece, slot_label))

        # 4. 根据BIO方式整理槽位实体，实际用到连续的B-、I-结构
        slots = {}               # 槽位名:值
        current_slot = None      # 当前采集中的属性名
        current_value = []       # 当前槽位的所有token片段
        for token, label in slot_tags:
            if label.startswith('B-'):
                # 完成上一个槽位，开始新槽位
                if current_slot and current_value:
                    slots[current_slot] = "".join(current_value)
                current_slot = label[2:]    # 槽位类别名
                current_value = [token]
            elif label.startswith('I-') and current_slot == label[2:]:
                # 同一个槽位延续
                current_value.append(token)
            else:
                # 非O结束/遇到O/断档，关闭当前槽位
                if current_slot and current_value:
                    slots[current_slot] = "".join(current_value)
                current_slot = None
                current_value = []
        # 句尾如仍有槽位未加
        if current_slot and current_value:
            slots[current_slot] = "".join(current_value)
        # 5. 最终结果返回，含意图label和全部槽位字典
        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots
        }