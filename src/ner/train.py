# @Time    : 2026/5/8 19:01
# @Author  : hero
# @File    : train.py
import os
import time
import evaluate

import datasets
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification, EvalPrediction, EarlyStoppingCallback
from configuration.config import *

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# 直接用huggingface帮我们实现的trainer


# 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 1 标签映射
id2label = {id: label for id, label in enumerate(LABELS)}
label2id = {label: id for id, label in enumerate(LABELS)}

# 2 模型
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# 3 加载数据集

# tips:因为之前用的是用save_to_disk,所以保存的是arrow,所以需要用load_from_disk
train_dataset = datasets.load_from_disk(
    PROCESSED_DATA_DIR / 'train'
)
valid_dataset = datasets.load_from_disk(
    PROCESSED_DATA_DIR / 'valid'
)

# 4 数据整理器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors='pt',
    label_pad_token_id=-100  # tips:作标签填充默认的tokenid
)

# 5 训练参数
log_dir = LOG_DIR / NER_DIR / time.strftime("%Y-%m-%d-%H-%M-%S")
log_dir.mkdir(parents=True, exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = str(log_dir)

args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR / NER_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=3,
    fp16=True,

    logging_strategy="steps",
    logging_steps=SAVE_STEPS,
    logging_first_step=True,
    report_to=["tensorboard"],  # tips：设置上报到tensorbord,这样日志就可以用tensorboard --logdir ./logs/ner通过webui查看了

    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    load_best_model_at_end=True,
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
        unpad_pred = [id2label[id] for id in unpad_pred]
        unpad_label = [id2label[id] for id in unpad_label]
        # 添加到列表
        unpad_labels.append(unpad_label)
        unpad_preds.append(unpad_pred)

    result = seqeval.compute(predictions=unpad_preds, references=unpad_labels)
    return result


# 7 tips 增加早停策略,可以在hf的transformerdoc的api章节查看https://huggingface.co/docs/transformers/v5.8.0/en/main_classes/callback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=20, #tips:设置早停机制,连续两次 eval（每20step一次）没提升就停，正式训练改为20,也就是连续20次eval没提升就停止
                                                # early_stopping_threshold: float | None = 0.0 #tips:设置早停阈值,譬如设置0.7,那么就是分数提升到0.7的时候就停止训练
)

# 创建训练器
trainer = Trainer(
    model=model,
    # args: TrainingArguments | None = None,
    args=args,  # tips:传入上面5设置的TrainingArguments
    # data_collator: (list[Any]) -> dict[str, Any] | None = None,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,

    # processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None = None,
    # model_init: (...) -> PreTrainedModel | None = None,
    # compute_loss_func: (...) -> Any | None = None,
    # compute_metrics: (EvalPrediction) -> dict | None = None,
    compute_metrics=compute_metrics,
    # callbacks: list[TrainerCallback] | None = None,
    callbacks=[early_stopping_callback], #tips：传入回调函数,设置早停法
    # optimizers: tuple[Optimizer | None, LambdaLR | None] = (None, None),
    # optimizer_cls_and_kwargs: tuple[type[Optimizer], dict[str, Any]] | None = None,
    # preprocess_logits_for_metrics: (Tensor, Tensor) -> Tensor | None = None)

)

# 训练
trainer.train()

# 模型保存
trainer.save_model(CHECKPOINT_DIR / NER_DIR / 'best_model')
