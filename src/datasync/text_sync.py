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