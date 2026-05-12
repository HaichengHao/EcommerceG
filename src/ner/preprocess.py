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