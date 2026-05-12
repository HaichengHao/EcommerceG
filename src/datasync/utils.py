# @Time    : 2026/5/12 09:20
# @Author  : hero
# @File    : utils.py
import pymysql
from pymysql.cursors import DictCursor
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



    # ==========================================================
    #下面是没有封装的时候的测试用代码,由于已经封装完毕,故注释掉
    # #定义Neo4j的driver
    # driver = GraphDatabase.driver(
    #     **NEO4J_CONFIG
    # )
    # # #测试读取cat
    # #
    # # #1.category1
    # # #1.1读取base_category1数据
    # # demosql="""
    # #         SELECT * FROM base_category1;
    # #     """
    # # category1 = reader.read(demosql)
    # # # print(category1)
    # # '''
    # # [{'id': 1, 'name': '图书、音像、电子书刊', 'create_time': datetime.datetime(2021, 12, 14, 0, 0), 'operate_time': None},
    # #  {'id': 2,......}
    # # ]
    # # '''
    # #
    # # #1.2 写入neo4j,标签是Category1
    # #
    # #
    # # '''for item in category1:
    # #     cypher_query="""
    # #         MERGE (n:category1{id:$id,name:$name})
    # #     """
    # #     driver.execute_query(
    # #         cypher_query,
    # #         # id=item['id'],
    # #         # name=item['name']
    # #         ## 由于item本身就是字典,那么可以换一种方式
    # #         parameters_=item  #tips:这样它就会去自动拿出item中的id和name给到query
    # #     )'''
    # #
    # # #但是这样显然效率很低下,python写循环并一个个传入neo4j,两者循环开销以及通信开销都大
    # # #下面引入neo4j的UNWIND用法,将列表直接发入neo4j,neo4j使用UNWIND拆分列表,将元素拆解,并对列表中的元素进行操作
    # # cypher_query="""
    # #     UNWIND $category1 AS item
    # #     MERGE (:category1{id:item.id,name:item.name})
    # #
    # # """
    # # # UNWIND $category1 AS item 相当于拿出category1列表中的每一个元素给它们编排为一列,然后就可以对列数据进行同一操作,
    # # # 利用MERGE (:category1{id:item.id,name:item.name})为item列的每一个元素创建节点
    # # driver.execute_query(
    # #     cypher_query,
    # #     category1=category1
    # # )
    # # logger.success('写入Neo4j成功')
    # #
    # # # ==============================
    # # # 2.category2
    # # # 2.1读取base_category2数据
    # # demosql = """ \
    # #           SELECT * \
    # #           FROM base_category2;
    # #         """
    # # category2 = reader.read(demosql)
    # #
    # #
    # # # 2.2 写入neo4j,标签是Category2
    # # cypher_query = """
    # #         UNWIND $category2 AS item
    # #         MERGE (:category2{id:item.id,name:item.name})
    # #
    # #     """
    # # driver.execute_query(
    # #     cypher_query,
    # #     category2=category2
    # # )
    # # logger.success('写入Neo4j成功')
    #
    # # 3创建节点关系 Category2->Belong->Category1
    # # 3.1 读取base_category1 数据
    # demosql = """
    #     SELECT id as start_id,category1_id as end_id FROM base_category2;
    # """
    # '''
    # 为什么这样写呢?因为base_category2第一条电子书刊category1_id=1对应base_category1中的1的图书、音像、电子书刊
    # 所以关系应该是bc2属于bc1大类,所以创建节点关系的时候就是(bc2)-[:Belong]->(bc1)
    # '''
    # relations = reader.read(demosql)
    # print(relations)
    # '''
    # [{'start_id': 1, 'end_id': 1}, {'start_id': 2, 'end_id': 1}, ...
    #
    # {'start_id': 45, 'end_id': 7}, {'start_id': 46, 'end_id': 7}]'''
    #
    # #3.2写入neo4j,标签Belong
    #
    # cypher="""
    #     UNWIND $relations AS item
    #     MATCH (start:category2 {id:item.start_id}),(end:category1 {id:item.end_id})
    #     MERGE (start)-[r:Belong]->(end)
    # """
    #
    #
    # driver.execute_query(
    #     cypher,
    #     relations=relations
    # )
    #
    # logger.success('写入关系成功!')
    #tips:测试完毕,封装为Neo4jWriter！！