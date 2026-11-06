"""
Generic retrieval-augmented generation helpers.

This module keeps the RAG wiring reusable while allowing individual apps to
provide their own knowledge-base and prompt configuration.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field

import chromadb
from google import genai
from google.genai import types

from conversation_memory import (
    BufferMemory,
    ConversationMemoryStrategy,
    NullMemory,
    SummaryBufferMemory,
)
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
    extract_pdf_images: bool = False
    # Diagram-aware retrieval: when chunks carry `content_type: "image"` metadata
    # (see VectorDBPopulator's "slide" chunking mode), re-attach the original
    # slide image to the generation call instead of only its text description.
    images_base_dir: str | None = None
    max_retrieved_images: int = 2


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


class GeminiSummarizer:
    """
    Callable that folds newly-evicted conversation lines into a running summary
    using a Gemini model. Injected into `SummaryBufferMemory` as its
    `summarize_fn`, keeping the memory strategies decoupled from the LLM SDK.
    """

    def __init__(self, api_key: str, model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __call__(self, existing_summary: str, new_lines: str) -> str:
        previous = existing_summary or "(none yet)"
        prompt = (
            "Progressively summarize the conversation below, extending the existing "
            "summary so it captures all important facts, decisions, and context. "
            "Return only the updated summary.\n\n"
            f"Existing summary: {previous}\n\n"
            f"New lines of conversation:\n{new_lines}\n\n"
            "Updated summary:"
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
        )
        return (response.text or existing_summary).strip()


def build_memory(
    api_key: str,
    llm_model_name: str,
    *,
    enable_memory: bool = False,
    memory_mode: str = "buffer",
    max_history_turns: int | None = None,
    summarization_model: str | None = None,
) -> ConversationMemoryStrategy:
    """
    Convenience factory translating friendly flags into a memory strategy.

    Presets (e.g. models/ExampleBot.py) use this to keep an ergonomic constructor
    while the core `GenericRAGChatbot` stays pure dependency-injection.
    """
    if not enable_memory:
        return NullMemory()
    if memory_mode == "summary":
        summarizer = GeminiSummarizer(api_key, summarization_model or llm_model_name)
        return SummaryBufferMemory(summarizer, max_history_turns)
    return BufferMemory(max_history_turns)


class GenericRAGChatbot:
    """Reusable RAG chatbot that can back different domain-specific apps."""

    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        api_key: str,
        knowledge_base: KnowledgeBaseConfig,
        prompt_config: PromptConfig | None = None,
        memory: ConversationMemoryStrategy | None = None,
    ):
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.knowledge_base = knowledge_base
        self.prompt_config = prompt_config or PromptConfig()
        self.chroma_client = chromadb.PersistentClient(path=knowledge_base.db_path)
        self.memory: ConversationMemoryStrategy = memory or NullMemory()

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

    def clear_history(self) -> None:
        """Discard all stored conversation history."""
        self.memory.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the current conversation history."""
        return self.memory.history

    def _collect_retrieved_image_parts(self, data) -> list:
        """Read the on-disk images for any retrieved diagram-slide chunks.

        Walks retrieval results in rank order, keeps `content_type == "image"`
        entries with a resolvable `image_path`, dedups by resolved path, and
        caps at `knowledge_base.max_retrieved_images`. Missing files are
        skipped with a warning rather than raised, so retrieval degrades to
        text-only instead of breaking the answer.
        """
        metadatas = ((data or {}).get("metadatas") or [[]])[0]
        max_images = self.knowledge_base.max_retrieved_images
        parts = []
        seen_paths = set()

        for metadata in metadatas:
            if len(parts) >= max_images:
                break
            if (metadata or {}).get("content_type") != "image":
                continue
            image_path = metadata.get("image_path")
            if not image_path:
                continue

            resolved_path = (
                os.path.join(self.knowledge_base.images_base_dir, image_path)
                if self.knowledge_base.images_base_dir
                else image_path
            )
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)

            try:
                with open(resolved_path, "rb") as f:
                    image_bytes = f.read()
            except OSError as e:
                print(f"Warning: could not read retrieved image {resolved_path}: {e}")
                continue

            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))

        return parts

    def _build_prompt(
        self,
        prompt: str,
        data,
        image_data: str | None = None,
        retrieved_image_count: int = 0,
    ) -> str:
        if isinstance(data, dict):
            data = self._format_retrieval_results(data)
        elif isinstance(data, list):
            data = " ".join(data)

        user_prompt = prompt or self.prompt_config.empty_prompt_fallback
        image_instruction = (
            f" {self.prompt_config.image_instruction}"
            if image_data or retrieved_image_count
            else ""
        )
        history_prefix = self.memory.render()

        return (
            f"{history_prefix}"
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
        retrieved_image_parts = self._collect_retrieved_image_parts(data)
        full_prompt = self._build_prompt(
            prompt,
            data,
            image_data=image_data,
            retrieved_image_count=len(retrieved_image_parts),
        )

        contents = []
        if image_data:
            mime = image_mime_type or "image/png"
            image_bytes = base64.b64decode(image_data)
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
        contents.extend(retrieved_image_parts)
        contents.append(full_prompt)

        response = self.client.models.generate_content(
            model=self.llm_model_name,
            contents=contents,
        )
        return response.text

    def update_vector_db(self, update_data_dirs: list[str] | None = None, load_types: list[str] | None = None) -> None:
        """Populate the configured vector store with fresh documents."""
        populator = VectorDBPopulator(
            api_key=self.api_key,
            embedding_model_name=self.embedding_model_name,
            collection_name=self.knowledge_base.collection_name,
            db_path=self.knowledge_base.db_path,
            chunking_mode=self.knowledge_base.chunking_mode,
            source_label=self.knowledge_base.source_label,
            extract_pdf_images=self.knowledge_base.extract_pdf_images, 
        )
        populator.populate(update_data_dirs or self.knowledge_base.data_dirs, load_types=load_types)

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
                    "Set UPDATE_CHROMA_DB = True in models/ExampleBot.py and run it "
                    "(see README 'Building the vector DB')."
                ) from e
            else:
                raise

        response_text = self._generate_response(
            prompt,
            data,
            image_data=image_data,
            image_mime_type=image_mime_type,
        )

        self.memory.add_turn(prompt or "", response_text)

        return response_text

    def retrieve_docs_only(self, prompt: str, k: int | None = None):
        """Retrieve relevant documents without generating a response."""
        return self._retrieve_relevant_documents(
            prompt,
            k=k or self.knowledge_base.retrieval_k,
        )