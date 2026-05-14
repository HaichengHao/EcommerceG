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






