# @Time    : 2026/5/8 19:01
# @Author  : hero
# @File    : evaluate.py
import os
import evaluate
from transformers import Trainer, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, EvalPrediction, EarlyStoppingCallback
import datasets
from configuration.config import *
#定义训练器

#定义用BEST_MODEL
BEST_MODEL=str(CHECKPOINT_DIR / NER_DIR / 'best_model')

# 1 分词器
tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL)


# 2 模型
model = AutoModelForTokenClassification.from_pretrained(
    BEST_MODEL,
)

# 3 加载数据集，只要测试集

# tips:因为之前用的是用save_to_disk,所以保存的是arrow,所以需要用load_from_disk
test_dataset = datasets.load_from_disk(
    PROCESSED_DATA_DIR / 'test'
)
valid_dataset = datasets.load_from_disk(
    PROCESSED_DATA_DIR / 'valid'
)

# 4 数据整理器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors='pt',
)



# 6.评估指标函数
seqeval = evaluate.load('seqeval')  # tips:可以在huggingface上面找evaluate


def compute_metrics(prediction: EvalPrediction):
    # 提取模型的预测输出和真实标签
    logits = prediction.predictions
    preds = logits.argmax(-1)  # 预测分类标签
    labels = prediction.label_ids  # 真实分类标签
    # 将标签id转换为真正的标注标签BIO
    unpad_labels = []
    unpad_preds = []
    for pred, label in zip(preds, labels):
        # 去掉填充对应的id，也就是之前为了匹配CLS标签和SEP标签填充的-100
        unpad_label = label[label != -100]
        unpad_pred = pred[label != -100]
        # 转BIO标签
        unpad_pred = [model.config.id2label[id] for id in unpad_pred]
        unpad_label = [model.config.id2label[id] for id in unpad_label]
        # 添加到列表
        unpad_labels.append(unpad_label)
        unpad_preds.append(unpad_pred)

    result = seqeval.compute(predictions=unpad_preds, references=unpad_labels)
    return result




#7 创建训练器
trainer = Trainer(
    model=model,
    # args: TrainingArguments | None = None,

    # data_collator: (list[Any]) -> dict[str, Any] | None = None,
    data_collator=data_collator,
    eval_dataset=valid_dataset,

    compute_metrics=compute_metrics,


)
#验证评估
result = trainer.evaluate(
)


#打印结果
print(result)

'''
/home/nikofox/.local/bin/uv run /home/nikofox/llm_projects/E-commerceMap/.venv/bin/python /home/nikofox/llm_projects/E-commerceMap/src/ner/eval.py 
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 9571.97it/s]
100%|██████████| 13/13 [00:00<00:00, 113.99it/s]
{'eval_loss': 0.6900218725204468, 
'eval_model_preparation_time': 0.0015,
'eval__': {'precision': 0.37988826815642457, 
'recall': 0.4610169491525424, 'f1': 0.4165390505359877, 'number': 295}, 
'eval_overall_precision': 0.37988826815642457, 
'eval_overall_recall': 0.4610169491525424, 
'eval_overall_f1': 0.4165390505359877, 针对ner任务而言,这个到0.4+就已经不错了
'eval_overall_accuracy': 0.7750730282375852, 针对BIO而言的,准确率会高很多
'eval_runtime': 0.247, 
'eval_samples_per_second': 404.847, 
'eval_steps_per_second': 52.63, 
'epoch': 0}

'''