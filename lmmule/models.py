from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    Computed,
    Text,
    String,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    author: Mapped[str | None] = mapped_column(String(255), index=True)
    type: Mapped[str] = mapped_column(String(50), index=True)  # pdf, web, book, etc.

    documents: Mapped[list["Document"]] = relationship(back_populates="source")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text)
    namespace: Mapped[str] = mapped_column(String(50), index=True)
    text_hash: Mapped[str] = mapped_column(
        String(32),
        Computed("md5(text::bytea)"),
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)

    source: Mapped["Source"] = relationship(back_populates="documents")

    __table_args__ = (
        UniqueConstraint("text_hash", "namespace", name="uq_text_hash_namespace"),
    )
