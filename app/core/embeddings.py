"""
embeddings.py —— Embedding（文本向量化）客户端

【这个文件做什么】
  把一段文本变成一串向量（数字列表），语义相近的文本向量距离更近。
  本项目走 OpenAI 兼容接口（embedding_api_key + embedding_base_url），
  因此 DeepSeek / 智谱 / 硅基流动等国内兼容服务都可以直接用。

【新手必读 · 面试点】
  "为什么向量检索能理解语义？"——因为 embedding 模型在海量语料上学过"语义相近 → 向量相近"。
  本项目不关心模型内部的数学，只把它当成一个"文本 → 向量"的函数使用。
"""
from openai import OpenAI

from app.config import settings


class OpenAICompatibleEmbedding:
    """封装 OpenAI 兼容接口的 embedding 调用。"""

    def __init__(self):
        # 只传有值的参数：如果没配 base_url，就默认 OpenAI 官方
        kwargs = {"api_key": settings.embedding_api_key}
        if settings.embedding_base_url:
            kwargs["base_url"] = settings.embedding_base_url
        self.client = OpenAI(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成向量。输入 list[str]，输出 list[list[float]]。

        ⚠️ 返回向量的维度必须等于 settings.embedding_dim，
        否则写进 Qdrant 时维度不匹配会报错——换模型时必须同步改 .env 里的 EMBEDDING_DIM。
        """
        response = self.client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_one(self, text: str) -> list[float]:
        """embed() 的单条便捷封装。"""
        return self.embed([text])[0]
