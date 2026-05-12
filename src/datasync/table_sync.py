# @Time    : 2026/5/12 16:06
# @Author  : hero
# @File    : table_sync.py

from utils import MysqlReader,Neo4jWriter

#构建一个表数据的同步器

class TableSynchronizer:
    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()

    #分类信息
    def sync_category1(self):
        sql = """
             SELECT id,name FROM base_category1
        """

        #读取mysql得到一组属性(id,name)列表
        properties = self.reader.read(sql)
        self.writer.write_nodes(label='Category1',properties=properties)