# @Time    : 2026/5/8 19:00
# @Author  : hero
# @File    : config.py
# from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()
#1.目录路径
ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
NER_DIR = 'ner'
RAW_DATA_DIR = DATA_DIR / NER_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / NER_DIR / "processed"

LOG_DIR = ROOT_DIR / "logs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

#2.数据文件名称和模型名称
RAW_DATA_FILE=str(RAW_DATA_DIR/ "data.json")
# MODEL_NAME='google-bert/bert-base-chinese'
MODEL_NAME='/home/nikofox/.cache/huggingface/hub/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea'


#3.本次适用HuggingFace的trainer,那么参数有一些就不需要自己设置了,这里只设置几个超参数
'''
本次数据只有1000条
由于LORA的秩过大以及batchsize过大会造成过拟合,所以batchsize要设置的小一些
'''
BATCH_SIZE=2  #计算公式 总迭代次数=E(总轮次数)*[N/B] (数据条数/批次数量) 每轮迭代次数=N/B
EPOCHS=5
LEARNING_RATE=5e-5

'''
算一下这样要迭代多少次,训练集总数据条数是800,B=2,那么每轮迭代次数为800/2=400次,由于设置的EPOCH为5,所以总迭代次数=5*800/2=2000'''

WHIGHT_DECAY=1e-4  #TIPS：设置权重衰减

SAVE_STEPS=20 #tips:控制每隔多少个训练步数（training steps）保存一次模型的 checkpoint（检查点）

#4.设置NER任务分类标签

LABELS=['B','I','O']

#
#
# class Settings(BaseSettings):
#     model_config = SettingsConfigDict(
#         env_file=ROOT_DIR / ".env",
#         extra=False
#     )
#

#5.数据库连接配置
MYSQL_CONFIG={
    'host':'127.0.0.1',
    'port':63306, #tips:docker本地映射mysql的3306
    'user':'root',
    'database':'gmall',
    'password':os.getenv('MYSQL_DOCKER'),

}
neo4jpwd=os.getenv('neo4j_pwd')
AUTH=("neo4j",neo4jpwd)
NEO4J_CONFIG={
    'uri':'neo4j://127.0.0.1:7687',
    'auth':AUTH
}