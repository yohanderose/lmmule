from sqlalchemy.orm import mapped_column
from sqlalchemy import text, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from pgvector.sqlalchemy import Vector
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys

from lmmule.mule import Mule, OPENROUTER_URL, OLLAMA_URL
from lmmule.models import Base, Source, Document


@dataclass
class EmbeddingProvider(ABC):
    model_name: str
    embed_dim: int

    @abstractmethod
    async def batch_embed(self, texts: list[str]) -> list[list[float]]:
        pass


@dataclass
class OllamaEmbedding(EmbeddingProvider):
    async def batch_embed(self, texts: list[str]) -> list[list[float]]:
        resp = await Mule.request(
            "POST",
            f"{OLLAMA_URL}/api/embed",
            payload={"model": self.model_name, "input": texts},
        )
        return resp["embeddings"] if resp.get("embeddings") else [[]]


@dataclass
class OpenRouterEmbedding(EmbeddingProvider):
    async def batch_embed(self, texts: list[str]) -> list[list[float]]:
        openrouter_key = Mule.get_openrouter_key()
        resp = await Mule.request(
            "POST",
            f"{OPENROUTER_URL}/embeddings",
            payload={"model": self.model_name, "input": texts},
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
        )
        return resp["embeddings"] if resp.get("embeddings") else [[]]


@dataclass
class Rag:
    postgres_url: str
    embedder: EmbeddingProvider

    def __post_init__(self):
        Document.embedding = mapped_column(Vector(self.embedder.embed_dim))

    async def init_db(self):
        self.engine = create_async_engine(self.postgres_url, echo=True)
        self.Session = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        if not await self.verify_db_connection():
            sys.exit()
        return self

    async def verify_db_connection(self) -> bool:
        try:
            async with self.Session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print("DB error:", e)
            return False

    async def upsert_source(
        self, name: str, author: str | None = None, type: str = "unknown"
    ) -> int:
        async with self.Session() as db:
            stmt = (
                insert(Source)
                .values(name=name, author=author, type=type)
                .on_conflict_do_update(
                    index_elements=["name"], set_=dict(author=author, type=type)
                )
                .returning(Source.id)
            )
            result = await db.execute(stmt)
            await db.commit()
            return result.scalar_one()

    async def upsert_documents(
        self,
        texts: list[str],
        source_id: int,
        namespace: str = "default",
        metadatas: list[dict] | None = None,
    ):
        metadatas = metadatas or [{}] * len(texts)
        embeddings = await self.embedder.batch_embed(texts)
        async with self.Session() as db:
            for text, embedding, metadata in zip(texts, embeddings, metadatas):
                stmt = (
                    insert(Document)
                    .values(
                        text=text,
                        namespace=namespace,
                        embedding=embedding,
                        source_id=source_id,
                        metadata_=metadata,
                    )
                    .on_conflict_do_nothing(constraint="uq_text_hash_namespace")
                )
                await db.execute(stmt)
            await db.commit()

    async def search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> list[dict]:
        query_embedding = (await self.embedder.batch_embed([query]))[0]
        async with self.Session() as db:
            stmt = (
                select(
                    Document.text,
                    Document.metadata_,
                    Source.name,
                    Source.author,
                    Source.type,
                    Document.embedding.cosine_distance(query_embedding).label(
                        "distance"
                    ),
                )
                .join(Document.source)
                .where(
                    Document.namespace == namespace,
                    Document.embedding.cosine_distance(query_embedding) < threshold,
                )
                .order_by("distance")
                .limit(top_k)
            )
            result = await db.execute(stmt)
            return [
                {
                    "text": text,
                    "metadata": metadata,
                    "source": {
                        "name": src_name,
                        "author": src_author,
                        "type": src_type,
                    },
                    "score": 1 - distance,
                }
                for text, metadata, src_name, src_author, src_type, distance in result
            ]

    async def get_all(self, namespace: str) -> list[dict]:
        async with self.Session() as db:
            stmt = select(Document).where(Document.namespace == namespace)
            result = await db.execute(stmt)
            return [
                {
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": doc.metadata_,
                    "source_id": doc.source_id,
                }
                for doc in result.scalars()
            ]
