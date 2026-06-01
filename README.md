# EcommerceG 

本项目旨在构建电商领域知识图谱，并基于该知识图谱搭建智能电商客服系统
![](/imgs/02.png)
## 技术栈

图数据库:Neo4j
关系型数据库:MySQL
深度学习框架:PyTorch
Web框架:FastAPI
大模型框架:LangChain



## NER 实体抽取
[实体抽取模块](/src/ner)

### 数据预处理
```python
# @Time    : 2026/5/8 19:00
# @Author  : hero
# @File    : preprocess.py
'''
数据预处理
这次要适用bertfortokenclassfication
要求传入
( input_ids: torch.Tensor | None = None
attention_mask:torch.Tensor | None = None
token_type_ids: torch.Tensor |None = None
position_ids: torch.Tensor | None = None
 inputs_embeds: torch.Tensor | None = Nonelabels: torch.Tensor |None = None
 **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] )
  → TokenClassifierOutput or tuple(torch.FloatTensor)

前三个期望输入直接通过tokenizer就可以,需要考虑后面参数的配置
'''

"""


回顾一下之前LLM_LEARN中Huggingface教程中的03
tokenizer = AutoTokenizer.from_pretrained("./pretrained/bert-base-chinese")
text = "我爱自然语言处理"

# 编码文本为模型输入格式
inputs = tokenizer(text)

print(inputs)

'''
{
'input_ids': [101, 2769, 4263, 5632, 4197, 6427, 6241, 1905, 4415, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}'''


# 除去text，tokenizer还提供了多个重要参数
texts = ["我爱自然语言处理", "我爱人工智能", "我们一起学习"]
inputs = tokenizer(
  texts,
  padding="max_length", # 自动补齐
  truncation=True, # 自动截断
  max_length=10, # 统一最大长度
  return_tensors="pt" # 返回 PyTorch 张量格式
)

print(inputs)

# 输出内容是一个包含三个字段的字典，每个字段是形状为 (batch_size, seq_len) 的张量
'''
{
	'input_ids': tensor([[ 101, 2769, 4263, 5632, 4197, 6427, 6241, 1905, 4415,  102],
                        [ 101, 2769, 4263,  782, 2339, 3255, 5543,  102,    0,    0],
                        [ 101, 2769,  812,  671, 6629, 2110,  739,  102,    0,    0]]), 
	'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
	'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
}'''


"""

from datasets import load_dataset
from transformers import AutoTokenizer
from configuration.config import *
from dotenv import load_dotenv
load_dotenv()
import os
# os.environ['HF_TOKEN']=os.getenv('')

def process():
    # 1.读取数据
    dataset = load_dataset('json', data_files=RAW_DATA_FILE)['train']
    print(dataset)

    # 2.去除多余的列(只需要text和label)
    dataset.remove_columns(['id', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'])

    # 3. 划分数据集,但是huggingface的train_test_split划分出的是一个训练集一个测试集,所以要想再划分,需要再划分
    dataset_dict = dataset.train_test_split(test_size=0.2)  # tips:训练集测试集8:2
    '''
    划分出来的返回结果就是DatasetDict对象
     DatasetDict({
            train: Dataset({
                features: ['text', 'label'],
                num_rows: 800
            })
            test: Dataset({
                features: ['text', 'label'],
                num_rows: 200
            })
        })
        '''
    dataset_dict['test'], dataset_dict['valid'] = dataset_dict['test'].train_test_split(
        test_size=0.5).values()  # tips:然后将测试集再划分,5,5开,划归验证集
    '''
    经过再次划分之后构建新的键valid,构建的验证集,然后现在DatasetDict就会是下面这样
    DatasetDict({
            train: Dataset({
                features: ['text', 'label'],
                num_rows: 800
            })
            test: Dataset({
                features: ['text', 'label'],
                num_rows: 100
            })
            valid: Dataset({
                features: ['text', 'label'],
                num_rows: 100
            })
    })
    
    '''
    # 4 定义分词器
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)

    # 5  利用分词器对原始数据中的text进行编码,还要把之前的TAG这样的label转换为BIO的Label

    def encode(example):
        # 5.0 如果想看到原始输入的text的话可以提前打印一下
        print(example['text'])
        # 5.1 将文本数据转成字符列表
        tokens = list(example['text']) #tips:"text": "麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩"
        # 5.2 文本编码(其实就是作一个convert_tokens_to_ids)
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,  # 自动截断
            # max_length=10,  # 统一最大长度
            # return_tensors="pt"  # 返回 PyTorch 张量格式
        )
        """
        tokenizer会返回类似于下面这样的
        {
            'input_ids': tensor([[ 101, 2769, 4263, 5632, 4197, 6427, 6241, 1905, 4415,  102],]), 
            'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
            'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],])
        }"""
        # 5.3 加上BIO标签,进行实体标注
        entities = example['label']
        # 定义标注列表，存放所有的id，默认都是O的id [B,I,O] ->[0,1,2]
        '''entities
        "label": [
          {
            "start": 3,
            "end": 7,
            "text": "德国进口",
            "labels": [
              "TAG"
            ]
          },
          {
            "start": 14,
            "end": 16,
            "text": "护眼",
            "labels": [
              "TAG"
            ]
          },
          {
            "start": 16,
            "end": 21,
            "text": "营养软胶囊",
            "labels": [
              "TAG"
            ]
          }
        ],
        这里的思路是,既然要打BIO标签,那么不如先将所有都变为O,然后只用关注B和I,对应上原始数据中的
        start打上B,然后start和end之间打上I就行了
        
        
        '''
        labels = [LABELS.index('O')] * len(tokens)
        # 遍历每个实体,也就是label列表
        for entity in entities:
            start = entity['start']
            end = entity['end']
            labels[start:end] = [LABELS.index('B')] + [LABELS.index('I')] * (end - start -1 )
            '''
            可以看一眼原始数据
            "text": "麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩",
            "id": 1,
            "label": [
              {
                "start": 3,
                "end": 7,
                "text": "德国进口",
                "labels": [
                  "TAG"
                ]
              },
              分析一下,我们把text的长度得到 len(text) = 30 
              然后创建了labels=[O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O]
              由于对应是索引位置,所以就是[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
              然后得到其中的start为3,end为7
              然后就将索引位置为3(3是开头begin,所以是B)-7(不包括7)的设置为B和I
              [O,O,O,B,I,I,I,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O]
              [2,2,2,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
              
              依次类推,第二个标签
              {
                "start": 14,
                "end": 16,
                "text": "护眼",
                "labels": [
                  "TAG"
                ]
              },
              start为14,end为16,那么把刚才[2,2,2,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
              变为[2,2,2,0,1,1,1,2,2,2,2,2,2,2,0,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
              但是还有一点需要注意,Bert是有起止标签的,也就是说它会在labels首尾加上<CLS>和<SEP>
            '''
        # important:前后加上id=-100,对应CLS和SEP
        labels=[-100]+labels+[-100] #tips:[-100,2,2,2,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,-100]
        inputs['labels']=labels
        return inputs
    # tips:利用map方法,将dataset_dict中的每个数据一个一个地都作自定义函数encode中的处理,
    dataset_dict = dataset_dict.map(encode, remove_columns=['text', 'label'])
    #tips:拿出第一条看看
    print(dataset_dict['train'][0])
    # important:为啥要删除呢？因为我们要把token转换为id,也就是input_ids
    # 标签（label）也要变成期待的labels
    # 处理之后原始的text,label作为原始列对于输入没用处了,所以可以利用remove_columns移除

    # 6 保存到文件
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)


if __name__ == '__main__':
    process()
"""
[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Dataset({
    features: ['text', 'id', 'label', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'],
    num_rows: 1000
})
Map: 100%|██████████| 800/800 [00:00<00:00, 4129.42 examples/s]
Map: 100%|██████████| 100/100 [00:00<00:00, 4341.17 examples/s]
Map: 100%|██████████| 100/100 [00:00<00:00, 3989.18 examples/s]
{'id': 29, 'annotator': 1, 'annotation_id': 49, 'created_at': '2025-09-08T15:26:42.427601Z', 
'updated_at': '2025-09-08T15:26:42.427601Z', 'lead_time': 38.505, 
'input_ids': [101, 5101, 1083, 5401, 2466, 1880, 5291, 143, 144, 4276, 5709, 7881, 3187, 5293, 2357, 1309, 2147, 1880, 5291, 741, 2791, 2145, 1324, 4371, 1068, 1874, 5344, 5682, 1870, 5291, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
'labels': [-100, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, -100]}
Saving the dataset (1/1 shards): 100%|██████████| 800/800 [00:00<00:00, 291980.79 examples/s]
Saving the dataset (1/1 shards): 100%|██████████| 100/100 [00:00<00:00, 39794.16 examples/s]
Saving the dataset (1/1 shards): 100%|██████████| 100/100 [00:00<00:00, 39756.44 examples/s]
"""
```

### 模型训练
```python
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

```


### 模型评估
```python
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


```


--------------------------------

## 知识图谱构建

利用neo4j+mysql实现从结构化数据中提取节点以及关系映射为Cypher并构建neo4j图数据库


### 数据同步工具# @Time    : 2026/5/12 16:06
# @Author  : hero
# @File    : table_sync.py

from utils import MysqlReader,Neo4jWriter
from loguru import logger
#构建一个表数据的同步器

class TableSynchronizer:
    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()

    #分类信息
    def sync_category1(self):
        sql = """
             SELECT id,name FROM base_category1;
        """

        #读取mysql得到一组属性(id,name)列表
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='Category1',properties=properties)
    def sync_category2(self):
        sql = """
             SELECT id,name FROM base_category2;
        """
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='Category2',properties=properties)

    def sync_category3(self):
        sql = """
              SELECT id, name \
              FROM base_category3; 
              """
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='Category3', properties=properties)

    #从下级分类表中提取与上级分类的关系
    def sync_category2_to_category1(self):
        sql = """
            SELECT id as start_id,\
            category1_id as end_id \
            FROM base_category2; 
            
        """
        relations = self.reader.read(sql)

        self.writer.write_relations(
            relationtype='Belong',
            start_label='Category2',
            end_label='Category1',
            relations=relations

        )

    def sync_category3_to_category2(self):
        sql = """
              SELECT id as start_id, \
              category2_id as end_id \
              FROM base_category3;

              """
        relations = self.reader.read(sql)

        self.writer.write_relations(
            relationtype='Belong',
            start_label='Category3',
            end_label='Category2',
            relations=relations

        )
    #================================================================================
    # 平台属性
    def sync_base_attr_name(self):
        sql = """
        SELECT id,attr_name as name FROM base_attr_info;
        """
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='BaseAttrName',properties=properties)

    def sync_base_attr_value(self):
        sql = """
            SELECT id,value_name as name from base_attr_value;
            """
        properties = self.reader.read(sql)
        self.writer.write_nodes(label="BaseAttrValue",properties=properties)

    def sync_base_attr_name_to_value(self):
        '''
        返回的是一个关系
        为什么这样写呢?可以看一眼,base_attr_value id为21的是RTX3070ti,是显卡,属于attr_id 26,
        然后看base_attr_info,id为26的是显卡
        所以应该是attr_id作为起始id,它have attr_value,即显卡中包含(Have)3070ti
        :return:
        '''
        sql = """
        SELECT id as end_id,attr_id as start_id FROM base_attr_value;
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="BaseAttrName",
            end_label="BaseAttrValue",
            relations=relations
        )
    def sync_category1_to_base_attr_name(self):
        sql = """
            SELECT category_id as start_id,id as end_id
            from base_attr_info
            where category_level=1"""
        relations = self.reader.read(sql)
        self.writer.write_relations(relationtype='Have',start_label="Category1",end_label="BaseAttrName",relations=relations)
    def sync_category2_to_base_attr_name(self):
        sql = """
            SELECT category_id as start_id,id as end_id
            from base_attr_info
            where category_level=2"""
        relations = self.reader.read(sql)
        self.writer.write_relations(relationtype='Have',start_label="Category2",end_label="BaseAttrName",relations=relations)
    def sync_category3_to_base_attr_name(self):
        sql = """
            SELECT category_id as start_id,id as end_id
            from base_attr_info
            where category_level=3"""
        relations = self.reader.read(sql)
        self.writer.write_relations(relationtype='Have',start_label="Category3",end_label="BaseAttrName",relations=relations)

    # ============================================================================================
    #商品信息和品牌信息

    #商品信息=============================
    def sync_spu(self):
        sql = """
            SELECT id,spu_name as name FROM spu_info;
        """
        properties = self.reader.read(sql)

        self.writer.write_nodes('SPU', properties=properties)

    def sync_sku(self):
        sql="""
            SELECT id,sku_name as name FROM sku_info;
        """

        properties = self.reader.read(sql)
        self.writer.write_nodes('SKU', properties=properties)

    def sync_sku_to_spu(self):
        sql="""
            SELECT id as start_id,spu_id as end_id
            FROM sku_info;
        """

        relations = self.reader.read(sql)

        self.writer.write_relations(
            relationtype='Belong',
            start_label="SKU",
            end_label="SPU",
            relations=relations
        )
    def sync_spu_to_category3(self):
        sql="""
        SELECT id as start_id,category3_id as end_id
        FROM spu_info;
        """

        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Belong',
            start_label='SPU',
            end_label='Category3',
            relations=relations
        )

    #品牌信息====================================================

    #定义品牌节点
    def sync_trademark(self):
        sql="""
            SELECT id,tm_name as name FROM base_trademark;
        """
        properties = self.reader.read(sql)
        self.writer.write_nodes('Trademark', properties=properties)

    #写入SPU节点和品牌节点之间的连接
    def sync_spu_to_trademark(self):
        sql="""
            SELECT id as start_id,tm_id as end_id FROM spu_info;
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Belong',
            start_label="SPU",
            end_label="Trademark",
            relations=relations
        )

    # 销售属性 ====================================================
    def sync_sale_attr_name(self):
        sql="""
            select id,sale_attr_name as name from spu_sale_attr;
        """
        properties=self.reader.read(sql)
        self.writer.write_nodes("SaleAttrName", properties=properties)
    def sync_sale_attr_value(self):
        sql="""
            select id,sale_attr_value_name as name from spu_sale_attr_value;
        """

        properties = self.reader.read(sql)
        self.writer.write_nodes("SaleAttrValue", properties=properties)

    def sync_sale_attr_name_to_value(self):
        sql="""
            SELECT a.id as start_id ,v.id as end_id \
            from spu_sale_attr as a join spu_sale_attr_value as v \
            on a.spu_id = v.spu_id 
            and a.base_sale_attr_id = v.base_sale_attr_id;
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="SaleAttrName",
            end_label="SaleAttrValue",
            relations=relations
        )
    def sync_spu_to_sale_attr_name(self):
        sql="""
        select spu_id as start_id,id as end_id
        from spu_sale_attr
        """
        relations =self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="SPU",
            end_label="SaleAttrName",
            relations=relations
        )
    def sync_sku_to_sale_attr_value(self):
        sql="""
            select sku_id as start_id,sale_attr_value_id as end_id
            from sku_sale_attr_value
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="SKU",
            end_label="SaleAttrValue",
            relations=relations
        )
    def sync_sku_to_base_attr_value(self):
        sql="""
            select sku_id as start_id,value_id as end_id
            from sku_attr_value
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="SKU",
            end_label="BaseAttrValue",
            relations=relations
        )

if __name__ == '__main__':

    #tips:同步分类数据
    synchronizer = TableSynchronizer()
    synchronizer.sync_category1()
    synchronizer.sync_category2()
    synchronizer.sync_category3()
    synchronizer.sync_category2_to_category1()
    synchronizer.sync_category3_to_category2()
    logger.success('同步分类数据完成!')

    #tips:同步平台属性
    synchronizer.sync_base_attr_name()
    synchronizer.sync_base_attr_value()
    synchronizer.sync_base_attr_name_to_value()
    synchronizer.sync_category1_to_base_attr_name()
    synchronizer.sync_category2_to_base_attr_name()
    synchronizer.sync_category3_to_base_attr_name()

    #tips:同步商品信息

    synchronizer.sync_spu()
    synchronizer.sync_sku()
    synchronizer.sync_sku_to_spu()
    synchronizer.sync_spu_to_category3()

    #tips:同步品牌信息
    synchronizer.sync_trademark()
    synchronizer.sync_spu_to_trademark()

    #tips:同步销售属性相关信息
    synchronizer.sync_sale_attr_name()
    synchronizer.sync_sale_attr_value()
    synchronizer.sync_sale_attr_name_to_value()
    synchronizer.sync_spu_to_sale_attr_name()
    synchronizer.sync_sku_to_sale_attr_value()
    synchronizer.sync_sku_to_base_attr_value()


    logger.success('全部数据同步完成')




```python
# @Time    : 2026/5/12 09:20
# @Author  : hero
# @File    : utils.py
import pymysql
from pymysql.cursors import DictCursor #important:用于将sql执行返回的数据转换为字典格式
from neo4j import GraphDatabase
from configuration.config import *
from loguru import logger

#定义mysql工具类

#创建mysql读取器
class MysqlReader:
    def __init__(self):
        self.conn = pymysql.connect(**MYSQL_CONFIG) #tips 直接解包赋值
        self.cursor = self.conn.cursor(DictCursor) # tips 使用DictCursor将返回值转换为字典类型

    # 查询mysql读取数据
    def read(self,sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    #关闭连接和游标
    def close(self):
        self.cursor.close()
        self.conn.close()

#创建neo4j写入器
class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)

    #定义写节点driver(批量写入,需要传入标签)
    def write_nodes(self,label:str,properties:list[dict]):
        #tips:因为传入的标签是需要动态变化的,但是MERGE中要传入的不能动态变化,所以需要用python中的字符串格式化了
        # 但是还有一点尴尬的是,f-string是花括号匹配,所以本身Cypher的属性字典也会被识别为占位符,那么如何解决呢?
        # 那就是在cypher中的属性字典外边再包一层花括号实现转义,这是f-string的设计规范⚠️
        cypher_query=f"""
            UNWIND $batch AS item
            MERGE (:{label} {{id:item.id,name:item.name}}) 

        """
        self.driver.execute_query(
            cypher_query,
            batch=properties
        )
    #定义写关系driver,传入关系类型
    def write_relations(self,relationtype:str,start_label,end_label,relations:list[dict]):
        cypher = f"""
                UNWIND $batch AS item  
                MATCH (start:{start_label}{{id:item.start_id}}),(end:{end_label} {{id:item.end_id}})
                MERGE (start)-[r:{relationtype}]->(end)
            """

        self.driver.execute_query(
            cypher,
            batch=relations
        )


if __name__ == '__main__':
    reader = MysqlReader()
    writer = Neo4jWriter()

    # 1.读取数据

    #1.category1
    #1.1读取base_category1数据
    demosql="""
            SELECT id,name FROM base_category1;
        """
    category1 = reader.read(demosql)
    # print(category1)
    # 1.2 写入neo4j,标签是Category1
    writer.write_nodes(label='Category1',properties=category1)

    # 2.category2
    # 2.1读取base_category2数据
    demosql = """ \
              SELECT id,name\
              FROM base_category2;
            """
    category2 = reader.read(demosql)
    # 2.2 写入neo4j,标签是Category2
    writer.write_nodes(
        label='Category2',properties=category2
    )
    logger.success('节点写入Neo4j成功')

    # 3创建节点关系 Category2->Belong->Category1
    # 3.1 读取base_category1 数据
    demosql = """
        SELECT id as start_id,category1_id as end_id FROM base_category2;
    """

    relations = reader.read(demosql)
    # print(relations)
    '''
    [{'start_id': 1, 'end_id': 1}, {'start_id': 2, 'end_id': 1}, ...

    {'start_id': 45, 'end_id': 7}, {'start_id': 46, 'end_id': 7}]'''

    #3.2写入neo4j,标签Belong


    writer.write_relations(
        relations=relations,
        start_label='Category2',
        end_label='Category1',
        relationtype='Belong',
    )

    logger.success('写入关系成功!')



```


### 同步结构化数据
```python
# @Time    : 2026/5/12 16:06
# @Author  : hero
# @File    : table_sync.py

from utils import MysqlReader,Neo4jWriter
from loguru import logger
#构建一个表数据的同步器

class TableSynchronizer:
    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()

    #分类信息
    def sync_category1(self):
        sql = """
             SELECT id,name FROM base_category1;
        """

        #读取mysql得到一组属性(id,name)列表
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='Category1',properties=properties)
    def sync_category2(self):
        sql = """
             SELECT id,name FROM base_category2;
        """
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='Category2',properties=properties)

    def sync_category3(self):
        sql = """
              SELECT id, name \
              FROM base_category3; 
              """
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='Category3', properties=properties)

    #从下级分类表中提取与上级分类的关系
    def sync_category2_to_category1(self):
        sql = """
            SELECT id as start_id,\
            category1_id as end_id \
            FROM base_category2; 
            
        """
        relations = self.reader.read(sql)

        self.writer.write_relations(
            relationtype='Belong',
            start_label='Category2',
            end_label='Category1',
            relations=relations

        )

    def sync_category3_to_category2(self):
        sql = """
              SELECT id as start_id, \
              category2_id as end_id \
              FROM base_category3;

              """
        relations = self.reader.read(sql)

        self.writer.write_relations(
            relationtype='Belong',
            start_label='Category3',
            end_label='Category2',
            relations=relations

        )
    #================================================================================
    # 平台属性
    def sync_base_attr_name(self):
        sql = """
        SELECT id,attr_name as name FROM base_attr_info;
        """
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='BaseAttrName',properties=properties)

    def sync_base_attr_value(self):
        sql = """
            SELECT id,value_name as name from base_attr_value;
            """
        properties = self.reader.read(sql)
        self.writer.write_nodes(label="BaseAttrValue",properties=properties)

    def sync_base_attr_name_to_value(self):
        '''
        返回的是一个关系
        为什么这样写呢?可以看一眼,base_attr_value id为21的是RTX3070ti,是显卡,属于attr_id 26,
        然后看base_attr_info,id为26的是显卡
        所以应该是attr_id作为起始id,它have attr_value,即显卡中包含(Have)3070ti
        :return:
        '''
        sql = """
        SELECT id as end_id,attr_id as start_id FROM base_attr_value;
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="BaseAttrName",
            end_label="BaseAttrValue",
            relations=relations
        )
    def sync_category1_to_base_attr_name(self):
        sql = """
            SELECT category_id as start_id,id as end_id
            from base_attr_info
            where category_level=1"""
        relations = self.reader.read(sql)
        self.writer.write_relations(relationtype='Have',start_label="Category1",end_label="BaseAttrName",relations=relations)
    def sync_category2_to_base_attr_name(self):
        sql = """
            SELECT category_id as start_id,id as end_id
            from base_attr_info
            where category_level=2"""
        relations = self.reader.read(sql)
        self.writer.write_relations(relationtype='Have',start_label="Category2",end_label="BaseAttrName",relations=relations)
    def sync_category3_to_base_attr_name(self):
        sql = """
            SELECT category_id as start_id,id as end_id
            from base_attr_info
            where category_level=3"""
        relations = self.reader.read(sql)
        self.writer.write_relations(relationtype='Have',start_label="Category3",end_label="BaseAttrName",relations=relations)

    # ============================================================================================
    #商品信息和品牌信息

    #商品信息=============================
    def sync_spu(self):
        sql = """
            SELECT id,spu_name as name FROM spu_info;
        """
        properties = self.reader.read(sql)

        self.writer.write_nodes('SPU', properties=properties)

    def sync_sku(self):
        sql="""
            SELECT id,sku_name as name FROM sku_info;
        """

        properties = self.reader.read(sql)
        self.writer.write_nodes('SKU', properties=properties)

    def sync_sku_to_spu(self):
        sql="""
            SELECT id as start_id,spu_id as end_id
            FROM sku_info;
        """

        relations = self.reader.read(sql)

        self.writer.write_relations(
            relationtype='Belong',
            start_label="SKU",
            end_label="SPU",
            relations=relations
        )
    def sync_spu_to_category3(self):
        sql="""
        SELECT id as start_id,category3_id as end_id
        FROM spu_info;
        """

        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Belong',
            start_label='SPU',
            end_label='Category3',
            relations=relations
        )

    #品牌信息====================================================

    #定义品牌节点
    def sync_trademark(self):
        sql="""
            SELECT id,tm_name as name FROM base_trademark;
        """
        properties = self.reader.read(sql)
        self.writer.write_nodes('Trademark', properties=properties)

    #写入SPU节点和品牌节点之间的连接
    def sync_spu_to_trademark(self):
        sql="""
            SELECT id as start_id,tm_id as end_id FROM spu_info;
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Belong',
            start_label="SPU",
            end_label="Trademark",
            relations=relations
        )

    # 销售属性 ====================================================
    def sync_sale_attr_name(self):
        sql="""
            select id,sale_attr_name as name from spu_sale_attr;
        """
        properties=self.reader.read(sql)
        self.writer.write_nodes("SaleAttrName", properties=properties)
    def sync_sale_attr_value(self):
        sql="""
            select id,sale_attr_value_name as name from spu_sale_attr_value;
        """

        properties = self.reader.read(sql)
        self.writer.write_nodes("SaleAttrValue", properties=properties)

    def sync_sale_attr_name_to_value(self):
        sql="""
            SELECT a.id as start_id ,v.id as end_id \
            from spu_sale_attr as a join spu_sale_attr_value as v \
            on a.spu_id = v.spu_id 
            and a.base_sale_attr_id = v.base_sale_attr_id;
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="SaleAttrName",
            end_label="SaleAttrValue",
            relations=relations
        )
    def sync_spu_to_sale_attr_name(self):
        sql="""
        select spu_id as start_id,id as end_id
        from spu_sale_attr
        """
        relations =self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="SPU",
            end_label="SaleAttrName",
            relations=relations
        )
    def sync_sku_to_sale_attr_value(self):
        sql="""
            select sku_id as start_id,sale_attr_value_id as end_id
            from sku_sale_attr_value
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="SKU",
            end_label="SaleAttrValue",
            relations=relations
        )
    def sync_sku_to_base_attr_value(self):
        sql="""
            select sku_id as start_id,value_id as end_id
            from sku_attr_value
        """
        relations = self.reader.read(sql)
        self.writer.write_relations(
            relationtype='Have',
            start_label="SKU",
            end_label="BaseAttrValue",
            relations=relations
        )

if __name__ == '__main__':

    #tips:同步分类数据
    synchronizer = TableSynchronizer()
    synchronizer.sync_category1()
    synchronizer.sync_category2()
    synchronizer.sync_category3()
    synchronizer.sync_category2_to_category1()
    synchronizer.sync_category3_to_category2()
    logger.success('同步分类数据完成!')

    #tips:同步平台属性
    synchronizer.sync_base_attr_name()
    synchronizer.sync_base_attr_value()
    synchronizer.sync_base_attr_name_to_value()
    synchronizer.sync_category1_to_base_attr_name()
    synchronizer.sync_category2_to_base_attr_name()
    synchronizer.sync_category3_to_base_attr_name()

    #tips:同步商品信息

    synchronizer.sync_spu()
    synchronizer.sync_sku()
    synchronizer.sync_sku_to_spu()
    synchronizer.sync_spu_to_category3()

    #tips:同步品牌信息
    synchronizer.sync_trademark()
    synchronizer.sync_spu_to_trademark()

    #tips:同步销售属性相关信息
    synchronizer.sync_sale_attr_name()
    synchronizer.sync_sale_attr_value()
    synchronizer.sync_sale_attr_name_to_value()
    synchronizer.sync_spu_to_sale_attr_name()
    synchronizer.sync_sku_to_sale_attr_value()
    synchronizer.sync_sku_to_base_attr_value()


    logger.success('全部数据同步完成')




```


### 同步非结构化数据
```python
# # @Time    : 2026/5/13 10:42
# # @Author  : hero
# # @File    : text_sync.py
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from configuration.config import *
from utils import MysqlReader, Neo4jWriter
from ner.predict import Predictor

class TextSynchronizer():
    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()
        # 定义一个实体的提取器，本质就是Predictor
        self.extractor = self._init_extractor()

    # 内部函数：初始化一个Predictor
    def _init_extractor(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model'))
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model'))

        return Predictor(model, tokenizer, device)

    # 同步Tag标签
    def sync_tag(self):
        # 1. 从MySQL提取商品描述信息
        sql = """
            select id, description
            from spu_info
        """
        spu_desc = self.reader.read(sql)#tips:因为utils中指定返回的是一个字典,这样的{'id':1,'description':'顶级拍照旗舰，影像效果超乎想象。'}
        # 但是模型期望的是一个[str,str,....]或者str,输出的将会是[['tag','tag'..],['tag','tag'..]]
        # 2. 拆分spu id 和 desc
        ids = [ item['id'] for item in spu_desc ]
        descs = [ item['description'] for item in spu_desc ]#tips:将description组装为一个列表 [顶级拍照旗舰，影像效果超乎想象。','高性价比全能手机，续航持久不卡顿。'...]

        # 3. 提取所有数据的 Tag 列表
        tags_list = self.extractor.extract(descs)

        # for id, tags in zip(ids, tags_list):
        #     print(id, tags)

        # 4. 构建Tag节点的属性（id,name），以及 SPU → Tag 关系（start_id，end_id）
        tag_properties = []  # tips:定义空列表用于存放tag
        relations=[] #tips:定义关系列表用于存放关系，这里忘记的话回看utils 第98行

        for id, tags in zip(ids, tags_list):#tips:([1,2,3,...],[['拍照旗舰','超乎想象'],['高性价比','不卡顿']....])
            # 遍历当前SPU的每个标签 #tips:id [1,2,3...] tags:[['拍照旗舰','超乎想象'],['高性价比','不卡顿']....]
            for index, tag in enumerate(tags):  #tips:(0,拍照旗舰)
                # 构建Tag属性
                tag_id = '-'.join([str(id), str(index)])#1-0 #tips:因为一个spu会有多个标签,所以我们给标签打上id的话就用spu_info的id以及
                property = {'id': tag_id, 'name': tag} #{'id':'1-0',name:'拍照旗舰'} ;{'id':'1-1',name:'超乎想象'}
                tag_properties.append(property) #[{'id':'1-0',name:'拍照旗舰'} ,{'id':'1-1',name:'超乎想象'}.....]
                # 构建关系
                relation = {'start_id': id, 'end_id': tag_id } #{'start_id': '1', 'end_id': '1-0' };{'start_id': '1', 'end_id':'1-1' }
                relations.append(relation) #[{'start_id': '1', 'end_id': '1-0' },{'start_id': '1', 'end_id':'1-1' },{'start_id': '2', 'end_id':'2-0' }... ]

        # 5. 写入Neo4j
        self.writer.write_nodes( "Tag", tag_properties) #tips:把标签节点写入
        self.writer.write_relations("Have", "SPU", "Tag", relations) #tips:再把SPU到Tag的关系(Have也就是SPU节点包含标签)写入


if __name__ == '__main__':

    synchronizer = TextSynchronizer()

    synchronizer.sync_tag()

    logger.success('标签同步完成!!')
```
--------------------------------------
## 知识图谱应用

### 创建索引
Neo4j的混合检索需要创建向量索引和全文索引,具体代码如下

```python
# @Time    : 2026/5/14 10:17
# @Author  : hero
# @File    : utils.py
import torch
# from accelerate.test_utils.scripts.external_deps.test_ds_alst_ulysses_sp import batch
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from neo4j_graphrag.types import SearchType

from configuration.config import *
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


class IndexUtil:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG['uri'],
            username=NEO4J_CONFIG['auth'][0],
            password=NEO4J_CONFIG['auth'][1]
        )

        # 定义，嵌入模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
            },
            encode_kwargs={
                'normalize_embeddings': True  # 要对向量做归一化
            }
        )

    # step 创建全文索引,传入索引名称,节点标签,属性
    # 关于OPTIONS直接在配置文件里修改了,
    # 将配置创建全文索引的配置(在neo4j/conf/neo4j.conf添加db.index.fulltext.default_analyzer=cjk),这里设置的是全文检索分析器

    '''
    https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/
    
    官方文档的示例
    CREATE FULLTEXT INDEX $name FOR (n:Employee|Manager) ON EACH [n.peerReviews]
    解读一下,这个就是为每个匹配到的节点(n:Employee|Manager)中的n.peerReviews创建一个名字叫作$name的索引
    
    OPTIONS {
      indexConfig: {
        `fulltext.analyzer`: 'english',
        `fulltext.eventually_consistent`: true
      }
    }
    
    '''

    def create_fulltext_index(self, index_name, label, property):
        #important:下面cypher的意思就是创建一个全文索引,名字你自己定,为谁创建呢?为标签名为label(你自己定)的节点的属性property(你自己定)创建
        cypher = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON EACH [n.{property}]
        """  # tips:还是强调,标签必须是固定的,所以无法通过$传递查询参数直接传入,只能通过格式化字符串传入,这里用f-string,但既然都是常量,那直接都改f-string

        self.graph.query(
            query=cypher
        )

    # ============================================================================

    # step 创建向量索引，需要传入生成向量的“源属性”，以及嵌入向量属性
    '''
    https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
    CREATE VECTOR INDEX moviePlots IF NOT EXISTS
    FOR (m:Movie)
    ON m.embedding ⚠️传入的是节点向量
    OPTIONS { indexConfig: {
     `vector.dimensions`: 1536,  ⚠️传入的是嵌入向量的维度
     `vector.similarity_function`: 'cosine'
    }}
    '''

    def create_vector_index(self, index_name, label, source_property, embedding_property):
        # 生成嵌入向量,并添加到节点属性中,思路是基于源属性生成向量
        embedding_dim = self._add_embedding(label, source_property, embedding_property) #important:先获取指定节点属性的向量，所以调用_add_embedding
        #important:下面cypher语句的意思就是为指定的节点中的属性(embedding_property)创建索引,名字自己定,要传入指定的属性的embedding向量
        cypher = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON n.{embedding_property}
            OPTIONS {{ indexConfig: {{
            `vector.dimensions`:{embedding_dim},
             `vector.similarity_function`: 'cosine'
            }}}}
            """
        self.graph.query(
            query=cypher
        )

    # tips:定义内部函数,生成嵌入向量并添加到节点属性中，返回向量维度
    def _add_embedding(self, label, source_property, embedding_property):
        # 1.查询所有节点对应的源属性值以及节点ID,作为模型的输入，还需要查出节点<id> important:注意是<id>,也就是每个节点的<id>它是唯一的,而不是不唯一的id
        #important: 从label类的节点中把对应的属性字段(source_property)都提取出来
        cypher = f"""
            MATCH (n:{label}) RETURN n.{source_property} as text,elementId(n) AS id; 
        """ #tips:用elementId(n)拿到节点n的ID
        results = self.graph.query(
            cypher
        )
        # 2.获取查询结果中的文本内容
        docs = [result['text'] for result in results]
        '''
        这样拿到的就是一个list[str]
        这样就可以用模型来embed了
        '''

        # 3.输入给embedding模型,得到嵌入向量
        embeddings = self.embedding_model.embed_documents(docs)
        # 4.将嵌入向量和id组合成字典形式
        batch = []
        # for id, vec in zip(results['id'], embeddings):
        #     batch.append({
        #         'id': id,
        #         'embedding': vec
        #     })
        for result, embedding in zip(results, embeddings):
            item = {'id': result['id'], 'embedding': embedding}
            batch.append(item)
        # 5.执行cypher,按id查节点,写入新的潜入向量属性
        cypher = f"""
            UNWIND $batch AS item
            MATCH (n:{label}) 
            WHERE elementId(n)=item.id
            SET n.{embedding_property} = item.embedding
                """
        self.graph.query(
            cypher,
            params={
                "batch":batch
            }
        )

        return len(embeddings[0])  # tips:拿到向量维度

    # tips:定义删除索引
    def drop_all_indexes(self):
        cypher = """
            SHOW indexes WHERE type IN ['VECTOR','FULLTEXT']
        """
        indexes = self.graph.query(
            cypher
        )
        for index in indexes:
            self.graph.query(
                f"DROP INDEX IF EXISTS {index}"
            )


if __name__ == '__main__':
    index = IndexUtil()

    #Trademark
    # 创建全文索引
    index.create_fulltext_index(
        index_name='trademark_fulltext_index',
        label='Trademark',
        property='name',
    )
    # # tips:扒拉一下create_fulltext_index的cypher
    # '''
    #  CREATE FULLTEXT INDEX trademark_fulltext_index IF NOT EXISTS
    #         FOR (n:Trademark) ON EACH [n.name]
    #
    #  意思就是为每一个Trademark标签的节点的name创建全文索引,索引名字叫作trademark_fulltext_index
    #
    # '''
    #
    # # 创建向量索引
    #
    index.create_vector_index(
        index_name='trademark_vector_index',
        label='Trademark',
        source_property='name',
        embedding_property='embedding',
    )
    #
    # logger.success('创建索引成功')
    #tips:嵌入模型
    # index_name = "trademark_vector_index"
    # keyword_index_name = "trademark_fulltext_index"
    # store = Neo4jVector.from_existing_index(
    #     index.embedding_model,
    #     url=NEO4J_CONFIG['uri'],
    #     username=NEO4J_CONFIG['auth'][0],
    #     password=NEO4J_CONFIG['auth'][1],
    #     index_name=index_name,
    #     keyword_index_name=keyword_index_name,
    #     search_type=SearchType.HYBRID,
    # )

    # result = store.similarity_search(
    #      'Apple',5
    # )
    # print(result)
#     [Document(metadata={}, page_content='苹果'), Document(metadata={}, page_content='华为'), Document(metadata={}, page_content='小米'), Document(metadata={}, page_content='VIVO'), Document(metadata={}, page_content='联想')]
#tips:测试成功了,接下来只要top-k为1找到最匹配的就行
    # result = store.similarity_search(
    #     'Apple', 1
    # )
    # print(result[0]['page_content'])

    # 这些测试完成之后,就可以创建所需要的索引了
    #SPU
    index.create_fulltext_index(
        index_name="spu_fulltext_index",
        label='SPU',
        property='name',

    )
    index.create_vector_index(
        index_name='spu_vector_index',
        label='SPU',
        source_property='name',
        embedding_property="embedding"
    )
    #SKU
    index.create_fulltext_index(
        index_name="sku_fulltext_index",
        label='SKU',
        property='name',

    )
    index.create_vector_index(
        index_name='sku_vector_index',
        label='SKU',
        source_property='name',
        embedding_property="embedding"
    )

    #category1
    index.create_fulltext_index(
        index_name="category1_fulltext_index",
        label='Category1',
        property='name',

    )
    index.create_vector_index(
        index_name='category1_vector_index',
        label='Category1',
        source_property='name',
        embedding_property="embedding"
    )
    # category2
    index.create_fulltext_index(
        index_name="category2_fulltext_index",
        label='Category2',
        property='name',

    )
    index.create_vector_index(
        index_name='category2_vector_index',
        label='Category2',
        source_property='name',
        embedding_property="embedding"
    )
    # category3
    index.create_fulltext_index(
        index_name="category3_fulltext_index",
        label='Category3',
        property='name',

    )
    index.create_vector_index(
        index_name='category3_vector_index',
        label='Category3',
        source_property='name',
        embedding_property="embedding"
    )

'''
https://docs.langchain.com/oss/python/integrations/vectorstores/neo4jvector#hybrid-search-vector-%2B-keyword
混合索引


写法很像之前写RAG构建向量数据库时传入的参数

#tips:一个是从没有索引会自行创建索引
# The Neo4jVector Module will connect to Neo4j and create a vector and keyword indices if needed.
hybrid_db = Neo4jVector.from_documents(
    docs,
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    search_type="hybrid",
)
另外一个是已经有索引的,可以把自己构造的索引也传入
index_name = "vector"  # default index name
keyword_index_name = "keyword"  # default keyword index name

store = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
    keyword_index_name=keyword_index_name,
    search_type="hybrid",
)
'''







```

### 创建聊天服务接口
```python
import os
import uuid
from datetime import datetime
from typing import Iterator

import redis
import torch
from dotenv import load_dotenv
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import ChatOpenAI
from neo4j_graphrag.types import SearchType

from configuration.config import EMBEDDING_MODEL, NEO4J_CONFIG

load_dotenv()


class ChatService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatOpenAI(
            model="glm-4.7",
            api_key=os.getenv("zhipu_key"),
            base_url=os.getenv("zhipu_base_url"),
        )

        self.neo4j_vectors = {
            "Trademark": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="trademark_vector_index",
                keyword_index_name="trademark_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "SPU": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="spu_vector_index",
                keyword_index_name="spu_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "SKU": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="sku_vector_index",
                keyword_index_name="sku_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "Category1": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="category1_vector_index",
                keyword_index_name="category1_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "Category2": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="category2_vector_index",
                keyword_index_name="category2_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
            "Category3": Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name="category3_vector_index",
                keyword_index_name="category3_fulltext_index",
                search_type=SearchType.HYBRID,
            ),
        }

        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

        self.redis_url = os.getenv("REDIS_DOCKER", "redis://127.0.0.1:6379/0")
        self.redis_key_prefix = "ecommerceg:chat:"
        self.redis_session_index_key = "ecommerceg:sessions"
        self.redis_session_title_key = "ecommerceg:session_titles"
        self.redis_session_ctime_key = "ecommerceg:session_ctimes"
        self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        self._memory_sessions: dict[str, dict] = {}
        self._memory_histories: dict[str, InMemoryChatMessageHistory] = {}

    def _redis_available(self) -> bool:
        try:
            return bool(self.redis_client.ping())
        except Exception:
            return False

    def _history(self, session_id: str):
        if self._redis_available():
            return RedisChatMessageHistory(
                session_id=session_id,
                url=self.redis_url,
                key_prefix=self.redis_key_prefix,
                ttl=None,
            )
        if session_id not in self._memory_histories:
            self._memory_histories[session_id] = InMemoryChatMessageHistory()
        return self._memory_histories[session_id]

    def create_session(self, title: str | None = None) -> dict:
        session_id = uuid.uuid4().hex
        created_at = datetime.now().isoformat(timespec="seconds")
        title_value = title or "新会话"

        if self._redis_available():
            self.redis_client.zadd(self.redis_session_index_key, {session_id: datetime.now().timestamp()})
            self.redis_client.hset(self.redis_session_title_key, session_id, title_value)
            self.redis_client.hset(self.redis_session_ctime_key, session_id, created_at)
        else:
            self._memory_sessions[session_id] = {
                "session_id": session_id,
                "title": title_value,
                "created_at": created_at,
                "last_score": datetime.now().timestamp(),
            }

        return {"session_id": session_id, "title": title_value, "created_at": created_at}

    def list_sessions(self) -> list[dict]:
        if self._redis_available():
            session_ids = self.redis_client.zrevrange(self.redis_session_index_key, 0, -1)
            titles = self.redis_client.hgetall(self.redis_session_title_key)
            ctimes = self.redis_client.hgetall(self.redis_session_ctime_key)

            result = []
            for sid in session_ids:
                result.append(
                    {
                        "session_id": sid,
                        "title": titles.get(sid, "新会话"),
                        "created_at": ctimes.get(sid, ""),
                    }
                )
            return result
        sessions = sorted(
            self._memory_sessions.values(),
            key=lambda x: x.get("last_score", 0),
            reverse=True,
        )
        return [
            {
                "session_id": item["session_id"],
                "title": item["title"],
                "created_at": item["created_at"],
            }
            for item in sessions
        ]

    def delete_session(self, session_id: str) -> None:
        if self._redis_available():
            self.redis_client.zrem(self.redis_session_index_key, session_id)
            self.redis_client.hdel(self.redis_session_title_key, session_id)
            self.redis_client.hdel(self.redis_session_ctime_key, session_id)
            self.redis_client.delete(f"{self.redis_key_prefix}{session_id}")
        self._memory_sessions.pop(session_id, None)
        self._memory_histories.pop(session_id, None)

    def get_session_messages(self, session_id: str) -> list[dict]:
        messages = self._history(session_id).messages
        result = []
        for m in messages:
            role = "assistant"
            if isinstance(m, HumanMessage):
                role = "user"
            result.append({"role": role, "content": m.content})
        return result

    def _build_history_context(self, session_id: str, max_turns: int = 6) -> str:
        messages = self._history(session_id).messages
        if not messages:
            return ""
        tail = messages[-(max_turns * 2):]
        lines = []
        for m in tail:
            if isinstance(m, HumanMessage):
                lines.append(f"用户: {m.content}")
            else:
                lines.append(f"助手: {m.content}")
        return "\n".join(lines)

    def chat_stream(self, question: str, session_id: str) -> Iterator[str]:
        history = self._history(session_id)
        history_context = self._build_history_context(session_id)

        result = self._generate_cypher(question)
        cypher = result["cypher_query"]
        entities_to_align = result["entities_to_align"]

        aligned_entities = self._entity_align(entities_to_align)
        query_result = self._execute_cypher(cypher, aligned_entities)

        full_answer = ""
        for chunk in self._generate_answer_stream(question, query_result, history_context):
            full_answer += chunk
            yield chunk

        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=full_answer))

        if self._redis_available():
            if self.redis_client.hget(self.redis_session_title_key, session_id) in (None, "新会话"):
                self.redis_client.hset(self.redis_session_title_key, session_id, question[:24])
            self.redis_client.zadd(self.redis_session_index_key, {session_id: datetime.now().timestamp()})
        elif session_id in self._memory_sessions:
            if self._memory_sessions[session_id]["title"] == "新会话":
                self._memory_sessions[session_id]["title"] = question[:24]
            self._memory_sessions[session_id]["last_score"] = datetime.now().timestamp()

    def _generate_cypher(self, question: str) -> dict:
        prompt = """
你是一个专业的Neo4j Cypher查询生成器。你的任务是根据用户问题生成一条Cypher查询语句，用于从知识图谱中获取回答用户问题所需的信息。

用户问题：{question}

知识图谱结构信息：{schema_info}

要求：
1. 生成参数化的Cypher查询语句，使用 $param_0, $param_1, ... 作为占位符
2. 识别需要对齐的实体，并记录原始名称和节点标签
3. 必须严格使用以下JSON格式输出结果：
{{
  "cypher_query": "生成的Cypher语句",
  "entities_to_align": [
    {{
      "param_name": "param_0",
      "entity": "原始实体名称",
      "label": "节点类型"
    }}
  ]
}}
"""
        template = PromptTemplate.from_template(prompt)
        rendered = template.format(question=question, schema_info=self.graph.schema)
        output = self.llm.invoke(rendered)
        return self.json_parser.invoke(output)

    def _entity_align(self, entities_to_align: list[dict]) -> list[dict]:
        for index, entity_to_align in enumerate(entities_to_align):
            label = entity_to_align["label"]
            entity = entity_to_align["entity"]
            aligned_entity = self.neo4j_vectors[label].similarity_search(entity, k=1)[0].page_content
            entities_to_align[index]["entity"] = aligned_entity
        return entities_to_align

    def _execute_cypher(self, cypher: str, aligned_entities: list[dict]):
        params = {aligned_entity["param_name"]: aligned_entity["entity"] for aligned_entity in aligned_entities}
        return self.graph.query(cypher, params=params)

    def _generate_answer_stream(self, question: str, query_result, history_context: str) -> Iterator[str]:
        prompt = """
你是一个电商智能客服，根据用户问题、会话历史和数据库查询结果，生成一段简洁、准确、连贯的自然语言回答。

会话历史：
{history_context}

用户问题: {question}
数据库返回结果: {query_result}
"""
        rendered = prompt.format(
            history_context=history_context or "(无历史)",
            question=question,
            query_result=query_result,
        )
        for chunk in self.llm.stream(rendered):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            yield text

```


### 接口定义
```python
import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from configuration.config import WEB_STATS_DIR
from schemas import CreateSessionRequest, Question, SessionQuestion
from service import ChatService

service = ChatService()

app = FastAPI(
    title="EcommerceG电商图谱助手",
    description="这是电商图谱助手后端接口",
)
app.mount("/static", StaticFiles(directory=WEB_STATS_DIR), name="static")


@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": service.list_sessions()}


@app.post("/api/sessions")
async def create_session(payload: CreateSessionRequest):
    return service.create_session(payload.title)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    service.delete_session(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    return {"messages": service.get_session_messages(session_id)}


@app.post("/api/chat")
async def chat_non_stream(question: Question):
    session = service.create_session("临时会话")

    def token_generator():
        yield from service.chat_stream(question.message, session["session_id"])

    full = "".join(list(token_generator()))
    return {"message": full}


@app.post("/api/chat/stream")
async def chat_stream(question: SessionQuestion):
    def token_generator():
        yield from service.chat_stream(question.message, question.session_id)

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

```

### schema定义
```python
from pydantic import BaseModel


class Question(BaseModel):
    message: str


class SessionQuestion(Question):
    session_id: str


class Answer(BaseModel):
    message: str


class CreateSessionRequest(BaseModel):
    title: str | None = None

```

