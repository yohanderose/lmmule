from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Computed, Text, String, text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.exc import IntegrityError
from pgvector.sqlalchemy import Vector
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

from lmmule.mule import Mule, OPENROUTER_URL, OLLAMA_URL


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
        class Base(DeclarativeBase):
            pass

        class Document(Base):
            __tablename__ = "documents"

            id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
            text: Mapped[str] = mapped_column(Text)
            text_hash: Mapped[str] = mapped_column(
                String(32),
                Computed("md5(text::bytea)"),
                unique=True,
            )
            # namespace: Mapped[str] = mapped_column(Text)
            # src_name: Mapped[str] = mapped_column(Text)
            # src_author: Mapped[str] = mapped_column(Text)
            # src_type: Mapped[str] = mapped_column(Text)
            embedding = mapped_column(Vector(self.embedder.embed_dim))
            metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

        self.Base = Base
        self.Document = Document

    async def init_db(self):
        self.engine = create_async_engine(self.postgres_url, echo=True)
        self.Session = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(self.Base.metadata.create_all)

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

    async def upsert_documents(self, texts: list[str], metadatas: list[dict] | None):
        metadatas = metadatas or [{}] * len(texts)
        embeddings = await self.embedder.batch_embed(texts)

        async with self.Session() as db:
            try:
                for text, embedding, metadata in zip(texts, embeddings, metadatas):
                    doc = self.Document(
                        text=text, embedding=embedding, metadata_=metadata
                    )

                    db.add(doc)
                    await db.commit()
            except IntegrityError:
                pass

    async def search(
        self, query: str, top_k: int = 5, threshold: float = 0.7
    ) -> list[dict]:
        query_embedding = (await self.embedder.batch_embed([query]))[0]
        async with self.Session() as db:
            stmt = (
                select(
                    self.Document.text,
                    self.Document.metadata_,
                    self.Document.embedding.cosine_distance(query_embedding).label(
                        "distance"
                    ),
                )
                .where(
                    self.Document.embedding.cosine_distance(query_embedding) < threshold
                )
                .order_by("distance")
                .limit(top_k)
            )

            result = await db.execute(stmt)
            return [
                {"text": text, "metadata": metadata, "score": 1 - distance}
                for text, metadata, distance in result
            ]
