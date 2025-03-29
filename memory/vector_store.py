import os
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import httpx

class VectorStore:
    """向量数据库接口，用于存储和检索向量化的记忆"""
    
    def __init__(self, collection_name: str = "memories", persist_directory: Optional[str] = None):
        """
        初始化向量数据库
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录，如果为None则使用内存存储
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "./chroma_db"
        
        # 创建Chroma客户端
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        
        # 选择嵌入函数
        self.embedding_function = self._get_embedding_function()
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
    
    def _get_embedding_function(self):
        """
        根据环境变量选择嵌入函数
        
        Returns:
            嵌入函数
        """
        # 尝试使用OpenRouter的嵌入API
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            try:
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openrouter_api_key,
                    model_name="text-embedding-ada-002",
                    api_base="https://openrouter.ai/api/v1"
                )
            except Exception as e:
                print(f"OpenRouter嵌入函数初始化失败: {e}")
        
        # 尝试使用DeepSeek的嵌入API
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            try:
                # 如果DeepSeek提供了与OpenAI兼容的嵌入API，可以使用这个
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=deepseek_api_key,
                    model_name="deepseek-embedding",
                    api_base="https://api.deepseek.com/v1"
                )
            except Exception as e:
                print(f"DeepSeek嵌入函数初始化失败: {e}")
        
        # 尝试使用OpenAI的嵌入API
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                # 尝试使用新的嵌入模型
                print("尝试使用OpenAI的text-embedding-3-small模型...")
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-3-small"
                )
            except Exception as e:
                print(f"OpenAI text-embedding-3-small初始化失败: {e}")
                try:
                    # 如果新模型失败，尝试使用旧模型
                    print("尝试使用OpenAI的text-embedding-ada-002模型...")
                    return embedding_functions.OpenAIEmbeddingFunction(
                        api_key=openai_api_key,
                        model_name="text-embedding-ada-002"
                    )
                except Exception as e:
                    print(f"OpenAI text-embedding-ada-002初始化失败: {e}")
                    print(f"错误类型: {type(e)}")
                    print(f"错误详情: {str(e)}")
        
        # 如果所有API都不可用，使用默认的嵌入函数
        print("警告: 所有嵌入API都不可用，使用默认的嵌入函数")
        print("注意: 默认嵌入函数性能较差，建议配置至少一个嵌入API")
        print("可用的嵌入API选项: OpenAI, DeepSeek, OpenRouter")
        return embedding_functions.DefaultEmbeddingFunction()
    
    async def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        添加记忆到向量数据库
        
        Args:
            text: 记忆文本
            metadata: 元数据
            id: 记忆ID，如果为None则自动生成
        
        Returns:
            记忆ID
        """
        try:
            memory_id = id or str(uuid.uuid4())
            
            # 确保元数据中的所有值都是字符串
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        metadata[key] = json.dumps(value)
                    elif not isinstance(value, str):
                        metadata[key] = str(value)
            
            print(f"正在添加记忆，ID: {memory_id}, 文本长度: {len(text)}")
            
            try:
                self.collection.add(
                    documents=[text],
                    metadatas=[metadata or {}],
                    ids=[memory_id]
                )
                print(f"记忆添加成功，ID: {memory_id}")
                return memory_id
            except Exception as e:
                print(f"添加记忆到向量数据库失败: {e}")
                print(f"错误类型: {type(e)}")
                print(f"错误详情: {str(e)}")
                
                # 检查是否是OpenAI API错误
                if "openai.NotFoundError" in str(type(e)):
                    print("OpenAI API错误: 模型或资源不存在")
                    print("请检查您的OpenAI API密钥和模型名称是否正确")
                    print("建议: 更新到最新的OpenAI嵌入模型，如text-embedding-3-small")
                
                # 尝试使用默认嵌入函数作为备选
                print("尝试使用默认嵌入函数作为备选...")
                try:
                    # 临时切换到默认嵌入函数
                    original_embedding_function = self.collection._embedding_function
                    self.collection._embedding_function = embedding_functions.DefaultEmbeddingFunction()
                    
                    self.collection.add(
                        documents=[text],
                        metadatas=[metadata or {}],
                        ids=[memory_id]
                    )
                    
                    print(f"使用默认嵌入函数添加记忆成功，ID: {memory_id}")
                    
                    # 恢复原始嵌入函数
                    self.collection._embedding_function = original_embedding_function
                    
                    return memory_id
                except Exception as backup_error:
                    print(f"使用默认嵌入函数添加记忆失败: {backup_error}")
                    raise
        except Exception as outer_error:
            print(f"添加记忆过程中发生未处理异常: {outer_error}")
            raise
    
    async def search_memories(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆
        
        Args:
            query: 查询文本
            filter: 过滤条件
            limit: 返回结果数量限制
        
        Returns:
            记忆列表，每个记忆包含id、text、metadata和distance字段
        """
        try:
            print(f"正在搜索记忆，查询: '{query}'")
            
            # 确保过滤条件中的所有值都是字符串
            if filter:
                for key, value in filter.items():
                    if isinstance(value, (dict, list)):
                        filter[key] = json.dumps(value)
                    elif not isinstance(value, str):
                        filter[key] = str(value)
                print(f"应用过滤条件: {filter}")
            
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=filter
                )
                print(f"搜索成功，找到 {len(results['documents'][0]) if results['documents'] and len(results['documents']) > 0 else 0} 条结果")
            except Exception as e:
                print(f"搜索记忆失败: {e}")
                print(f"错误类型: {type(e)}")
                print(f"错误详情: {str(e)}")
                
                # 检查是否是OpenAI API错误
                if "openai.NotFoundError" in str(type(e)):
                    print("OpenAI API错误: 模型或资源不存在")
                    print("请检查您的OpenAI API密钥和模型名称是否正确")
                    print("建议: 更新到最新的OpenAI嵌入模型，如text-embedding-3-small")
                
                # 尝试使用默认嵌入函数作为备选
                print("尝试使用默认嵌入函数作为备选...")
                try:
                    # 临时切换到默认嵌入函数
                    original_embedding_function = self.collection._embedding_function
                    self.collection._embedding_function = embedding_functions.DefaultEmbeddingFunction()
                    
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=limit,
                        where=filter
                    )
                    
                    print(f"使用默认嵌入函数搜索成功，找到 {len(results['documents'][0]) if results['documents'] and len(results['documents']) > 0 else 0} 条结果")
                    
                    # 恢复原始嵌入函数
                    self.collection._embedding_function = original_embedding_function
                except Exception as backup_error:
                    print(f"使用默认嵌入函数搜索失败: {backup_error}")
                    # 如果备选方案也失败，返回空结果
                    return []
            
            memories = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    memories.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                    })
            
            return memories
        except Exception as outer_error:
            print(f"搜索记忆过程中发生未处理异常: {outer_error}")
            # 返回空结果
            return []
    
    async def get_memory(self, id: str) -> Optional[Dict[str, Any]]:
        """
        获取记忆
        
        Args:
            id: 记忆ID
        
        Returns:
            记忆，包含id、text和metadata字段，如果不存在则返回None
        """
        try:
            print(f"正在获取记忆，ID: {id}")
            
            try:
                result = self.collection.get(ids=[id])
                
                if result["documents"] and len(result["documents"]) > 0:
                    print(f"记忆获取成功，ID: {id}")
                    return {
                        "id": id,
                        "text": result["documents"][0],
                        "metadata": result["metadatas"][0] if result["metadatas"] else {}
                    }
                print(f"记忆不存在，ID: {id}")
                return None
            except Exception as e:
                print(f"获取记忆失败: {e}")
                print(f"错误类型: {type(e)}")
                print(f"错误详情: {str(e)}")
                return None
        except Exception as outer_error:
            print(f"获取记忆过程中发生未处理异常: {outer_error}")
            return None
    
    async def update_memory(
        self, 
        id: str, 
        text: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新记忆
        
        Args:
            id: 记忆ID
            text: 新的记忆文本，如果为None则不更新
            metadata: 新的元数据，如果为None则不更新
        
        Returns:
            是否成功
        """
        try:
            # 获取现有记忆
            existing_memory = await self.get_memory(id)
            if not existing_memory:
                return False
            
            # 准备更新数据
            update_text = text if text is not None else existing_memory["text"]
            update_metadata = metadata if metadata is not None else existing_memory["metadata"]
            
            # 确保元数据中的所有值都是字符串
            if update_metadata:
                for key, value in update_metadata.items():
                    if isinstance(value, (dict, list)):
                        update_metadata[key] = json.dumps(value)
                    elif not isinstance(value, str):
                        update_metadata[key] = str(value)
            
            # 更新记忆
            self.collection.update(
                ids=[id],
                documents=[update_text],
                metadatas=[update_metadata]
            )
            
            return True
        except Exception as e:
            print(f"更新记忆失败: {e}")
            return False
    
    async def delete_memory(self, id: str) -> bool:
        """
        删除记忆
        
        Args:
            id: 记忆ID
        
        Returns:
            是否成功
        """
        try:
            self.collection.delete(ids=[id])
            return True
        except Exception as e:
            print(f"删除记忆失败: {e}")
            return False
    
    async def delete_memories(self, ids: List[str]) -> bool:
        """
        批量删除记忆
        
        Args:
            ids: 记忆ID列表
        
        Returns:
            是否成功
        """
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception as e:
            print(f"批量删除记忆失败: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "count": count
            }
        except Exception as e:
            print(f"获取集合统计信息失败: {e}")
            return {
                "collection_name": self.collection_name,
                "count": 0,
                "error": str(e)
            }