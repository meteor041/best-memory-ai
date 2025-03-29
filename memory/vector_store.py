import os
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple, Union
import json

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
        
        # 使用OpenAI的嵌入函数
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        
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
        memory_id = id or str(uuid.uuid4())
        
        # 确保元数据中的所有值都是字符串
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    metadata[key] = json.dumps(value)
                elif not isinstance(value, str):
                    metadata[key] = str(value)
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[memory_id]
        )
        
        return memory_id
    
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
        # 确保过滤条件中的所有值都是字符串
        if filter:
            for key, value in filter.items():
                if isinstance(value, (dict, list)):
                    filter[key] = json.dumps(value)
                elif not isinstance(value, str):
                    filter[key] = str(value)
        
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=filter
        )
        
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
    
    async def get_memory(self, id: str) -> Optional[Dict[str, Any]]:
        """
        获取记忆
        
        Args:
            id: 记忆ID
        
        Returns:
            记忆，包含id、text和metadata字段，如果不存在则返回None
        """
        try:
            result = self.collection.get(ids=[id])
            
            if result["documents"] and len(result["documents"]) > 0:
                return {
                    "id": id,
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0] if result["metadatas"] else {}
                }
            return None
        except Exception as e:
            print(f"获取记忆失败: {e}")
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