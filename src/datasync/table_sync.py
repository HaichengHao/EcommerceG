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



