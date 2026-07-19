import asyncio
from typing import Optional

from langchain_core.embeddings import Embeddings
from huggingface_hub import AsyncInferenceClient, InferenceClient

from app.conf.app_config import EmbeddingConfig, app_config


class _TEIEmbeddings(Embeddings):
    def __init__(self, url: str):
        self._client = InferenceClient(model=url)
        self._async_client = AsyncInferenceClient(model=url)

    def _format(self, texts: list[str]) -> list[str]:
        return [t.replace("\n", " ") for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.feature_extraction(text=self._format(texts)).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        r = await self._async_client.feature_extraction(text=self._format(texts))
        return r.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]


class EmbeddingClientManager:
    def __init__(self, config: EmbeddingConfig):
        self.client: Optional[_TEIEmbeddings] = None
        self.config = config

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        self.client = _TEIEmbeddings(self._get_url())


# 模块级单例，供其他模块按需复用同一个客户端管理器
embedding_client_manager = EmbeddingClientManager(app_config.embedding)


if __name__ == "__main__":
    # 本地调试入口：初始化客户端后执行一次最小化向量化调用
    embedding_client_manager.init()
    client = embedding_client_manager.client

    async def test():
        # 使用示例文本验证 Embedding 服务是否可正常响应
        text = "What is deep learning?"
        query_result = await client.aembed_query(text)
        # 只打印前 3 个维度，便于快速确认返回结果结构正确
        print(query_result[:3])

    # 运行调试测试
    asyncio.run(test())
