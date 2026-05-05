"""
Generic retrieval-augmented generation helpers.

This module keeps the RAG wiring reusable while allowing individual apps to
provide their own knowledge-base and prompt configuration.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import chromadb
from google import genai
from google.genai import types

from vector_db_populator import VectorDBPopulator


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """Configuration for a retrievable knowledge base."""

    collection_name: str
    db_path: str
    data_dirs: list[str] = field(default_factory=list)
    source_label: str = "Knowledge Base Material"
    chunking_mode: str = "recursive"
    fallback_search_query: str = "general knowledge question"
    retrieval_k: int = 7


@dataclass(frozen=True)
class PromptConfig:
    """Prompt shaping for a generic RAG assistant."""

    assistant_role: str = "a helpful assistant"
    source_description: str = "the supplied knowledge base"
    answer_style: str = "Keep the response concise and grounded in the retrieved context."
    citation_instruction: str = "Cite specific file names or source labels where possible."
    image_instruction: str = (
        "If an image is attached, analyze it together with the retrieved context before answering."
    )
    empty_prompt_fallback: str = "What can you tell me about this image?"


class GenericRAGChatbot:
    """Reusable RAG chatbot that can back different domain-specific apps."""

    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        api_key: str,
        knowledge_base: KnowledgeBaseConfig,
        prompt_config: PromptConfig | None = None,
    ):
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.knowledge_base = knowledge_base
        self.prompt_config = prompt_config or PromptConfig()
        self.chroma_client = chromadb.PersistentClient(path=knowledge_base.db_path)

    def get_collection_count(self) -> int:
        """Return the number of rows in the configured collection."""
        collection = self.chroma_client.get_collection(
            name=self.knowledge_base.collection_name
        )
        return collection.count()

    def _retrieve_relevant_documents(self, prompt: str, k: int = 1):
        """Retrieve the most relevant documents from the vector store."""
        collection = self.chroma_client.get_collection(
            name=self.knowledge_base.collection_name
        )

        embed_response = self.client.models.embed_content(
            model=self.embedding_model_name,
            contents=prompt,
        )
        embedding_vector = embed_response.embeddings[0].values

        return collection.query(
            query_embeddings=[embedding_vector],
            n_results=k,
        )

    @staticmethod
    def _format_retrieval_results(results) -> str:
        """Turn a Chroma query response into a readable context block."""
        documents = (results or {}).get("documents") or []
        metadatas = (results or {}).get("metadatas") or []

        if not documents:
            return "No relevant context was retrieved."

        context_chunks = []
        for query_index, query_documents in enumerate(documents):
            query_metadatas = metadatas[query_index] if query_index < len(metadatas) else []
            for doc_index, doc_text in enumerate(query_documents):
                metadata = query_metadatas[doc_index] if doc_index < len(query_metadatas) else {}
                source_name = metadata.get("file_name") or metadata.get("source") or f"Document {doc_index + 1}"
                context_chunks.append(
                    f"Source: {source_name}\n{doc_text.strip()}"
                )

        return "\n\n".join(context_chunks)

    def _build_prompt(self, prompt: str, data, image_data: str | None = None) -> str:
        if isinstance(data, dict):
            data = self._format_retrieval_results(data)
        elif isinstance(data, list):
            data = " ".join(data)

        user_prompt = prompt or self.prompt_config.empty_prompt_fallback
        image_instruction = (
            f" {self.prompt_config.image_instruction}" if image_data else ""
        )

        return (
            f"Using this data from {self.prompt_config.source_description}: {data}. "
            f"Respond as {self.prompt_config.assistant_role}. "
            f"{self.prompt_config.answer_style} "
            f"{self.prompt_config.citation_instruction}{image_instruction} "
            f"Answer this now: {user_prompt}"
        )

    def _generate_response(
        self,
        prompt: str,
        data,
        image_data: str | None = None,
        image_mime_type: str | None = None,
    ) -> str:
        full_prompt = self._build_prompt(prompt, data, image_data=image_data)

        contents = []
        if image_data:
            mime = image_mime_type or "image/png"
            image_bytes = base64.b64decode(image_data)
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
        contents.append(full_prompt)

        response = self.client.models.generate_content(
            model=self.llm_model_name,
            contents=contents,
        )
        return response.text

    def update_vector_db(self, update_data_dirs: list[str] | None = None) -> None:
        """Populate the configured vector store with fresh documents."""
        populator = VectorDBPopulator(
            api_key=self.api_key,
            embedding_model_name=self.embedding_model_name,
            collection_name=self.knowledge_base.collection_name,
            db_path=self.knowledge_base.db_path,
            chunking_mode=self.knowledge_base.chunking_mode,
            source_label=self.knowledge_base.source_label,
        )
        populator.populate(update_data_dirs or self.knowledge_base.data_dirs)

    def run(
        self,
        prompt: str,
        image_data: str | None = None,
        image_mime_type: str | None = None,
    ) -> str:
        search_query = prompt or self.knowledge_base.fallback_search_query
        try:
            data = self._retrieve_relevant_documents(
                search_query,
                k=self.knowledge_base.retrieval_k,
            )
        except ValueError as e:
            # chromadb raises ValueError("Collection <name> does not exist.")
            msg = str(e)
            missing_msg = f"Collection {self.knowledge_base.collection_name} does not exist."
            if missing_msg in msg:
                raise RuntimeError(
                    f"Vector DB collection '{self.knowledge_base.collection_name}' not found. "
                    "Build the vector database before querying.\n"
                    "Options: 1) Run the helper script 'build_db.py' (see README 'Building the vector DB'), "
                    "or 2) set UPDATE_CHROMA_DB = True in WineBot.py and run it to populate 'fnh330_db/'."
                ) from e
            else:
                raise

        return self._generate_response(
            prompt,
            data,
            image_data=image_data,
            image_mime_type=image_mime_type,
        )

    def retrieve_docs_only(self, prompt: str, k: int | None = None):
        """Retrieve relevant documents without generating a response."""
        return self._retrieve_relevant_documents(
            prompt,
            k=k or self.knowledge_base.retrieval_k,
        )