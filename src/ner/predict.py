# @Time    : 2026/5/8 19:01
# @Author  : hero
# @File    : predict.py
import torch
from markdown_it.common.entities import entities
# from torchaudio.models.squim.subjective import Predictor
from transformers import AutoTokenizer, AutoModelForTokenClassification
from configuration.config import *


class Predictor:
    # tips:初始化
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer = tokenizer

    # tips:预测方法
    def predict(self, inputs: str | list[str]):
        # tips:如果是一条数据,转换成列表处理
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]

        # 1. 预分词,得到字符列表
        tokens_lst = [list(input) for input in inputs]
        # 2. 用分词器对tokens_lst 进行id化处理
        input_tensor = self.tokenizer(
            tokens_lst,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # 3.加载到设备上
        input_tensor = {k: v.to(self.device) for k, v in input_tensor.items()}
        # 4.前向传播,推理预测
        with torch.no_grad():
            outputs = self.model(**input_tensor)  # tips:outputs是输出的结果,这里就是前向传播
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).tolist()

        # 5.将id列表转换成BIO标签
        final_predictions = []
        for tokens, prediction in zip(tokens_lst, predictions):
            # 截取预测输出中真实的长度
            prediction = prediction[1:len(tokens) + 1]
            # 转换成标签
            final_prediction = [self.model.config.id2label[id] for id in prediction]
            final_predictions.append(final_prediction)

        if is_str:
            return final_predictions[0]
        return final_predictions

    # tips:增加实体抽取方法
    def extract(self, inputs: str | list[str]):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        # 得到预测标签列表
        predictions = self.predict(inputs)
        # 从当前列表中抽取实体列表
        entities_list = []
        for input, labels in zip(inputs, predictions):
            # 调用内部函数,抽取一个数据样本的所有实体标签
            entities = self._extract_entities(list(input), labels)
            entities_list.append(entities)
        if is_str:
            return entities_list[0]
        return entities_list
    def _extract_entities(self,tokens,labels):
        entities=[]
        current_entity=""
        for token,label in zip(tokens,labels):
            if label == 'B': #开始保存新的实体
                if current_entity:
                    entities.append(current_entity)
                current_entity=token
            # 如果标签是I
            elif label == 'I':
                if current_entity:
                    current_entity+=token

            # 如果标签是O,就将实体抽取出来(如果存在),添加到列表
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity=""
        if current_entity:
            entities.append(current_entity)
        return entities


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model'))
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model'))
    # 定义预测器
    predictor = Predictor(model, tokenizer, device)

    # 定义数据
    # text = ["麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩","pvc水晶板软玻璃桌布防水桌子垫子透明加厚防烫塑料茶几垫餐桌垫"]

    # 测试预测
    # result = predictor.predict(text)
    #
    # for token, label in zip(text, result):
    #     print(token, label)

    #tips:实体抽取方法_extract_entities写好了,直接用它来做实体抽取
    # entities = predictor.extract(text)
    # print(entities)
    return predictor

if __name__ == '__main__':
    predict()

'''
麦 O
德 O
龙 O
德 B
国 I
进 I
口 I
双 O
心 O
多 O
维 O
叶 O
黄 O
素 O
护 B
眼 I
营 B
养 I
软 I
胶 I
囊 I
3 O
0 O
粒 O
x O
3 O
盒 O
眼 O
干 O
涩 O
'''
