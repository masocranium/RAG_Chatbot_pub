"""
Handles loading documents, chunking, and populating the ChromaDB vector store.
"""
import os
import sys
import hashlib
import math
import time
import chromadb
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document

CHARS_PER_TOKEN_ESTIMATE = 4


class CustomTextLoader(TextLoader):
    """Custom text loader to handle encoding issues."""

    def __init__(self, file_path, encoding="utf-8", autodetect_encoding=False):
        super().__init__(file_path, encoding, autodetect_encoding)

    def lazy_load(self):
        with open(self.file_path, "r", encoding=self.encoding, errors="ignore") as f:
            text = f.read()
        yield Document(
            page_content=text,
            metadata={"file_name": os.path.basename(self.file_path)},
        )


class VectorDBPopulator:
    """
    Loads documents from directories, chunkifies them (recursive or semantic),
    and stores them in a ChromaDB collection using Google AI embeddings.
    """

    def __init__(
        self,
        api_key: str,
        embedding_model_name: str,
        collection_name: str,
        db_path: str,
        chunking_mode: str = "recursive",
        source_label: str = "Knowledge Base Material",
    ):
        """
        Args:
            api_key: Google API key for embeddings.
            embedding_model_name: Model name for embed_content (e.g. gemini-embedding-001).
            collection_name: ChromaDB collection name.
            db_path: Path to the ChromaDB persistence directory.
            chunking_mode: "recursive" or "semantic".
        """
        genai.configure(api_key=api_key)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.db_path = db_path
        self.chunking_mode = chunking_mode
        self.source_label = source_label
        self.client = chromadb.PersistentClient(path=db_path)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count from character length for progress display."""
        return max(0, len(text) // CHARS_PER_TOKEN_ESTIMATE)

    def _load_documents(self, directory: str) -> list:
        """Load raw documents from a directory (txt and pdf)."""
        documents = []
        if not os.path.exists(directory):
            return documents
        text_loader = DirectoryLoader(
            directory, glob="**/*.txt", loader_cls=CustomTextLoader
        )
        try:
            documents.extend(text_loader.load())
        except Exception as e:
            print(f"Error loading text files: {e}", file=sys.stderr)
        pdf_loader = DirectoryLoader(
            directory, glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        try:
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error loading PDF files: {e}", file=sys.stderr)
        return documents

    def _chunk_documents(self, documents: list) -> list:
        """Apply the configured chunking strategy."""
        if self.chunking_mode == "semantic":
            return self._semantic_chunk_documents(documents)
        return self._recursive_chunk_documents(documents)

    def _recursive_chunk_documents(
        self,
        documents: list,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> list:
        """Character-based recursive chunking with progress."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        total_docs = len(documents)
        all_chunks = []
        total_tokens_est = 0
        for i, doc in enumerate(documents):
            est_tokens = self._estimate_tokens(doc.page_content)
            total_tokens_est += est_tokens
            print(
                f"\r  Recursive chunking: document {i + 1}/{total_docs} "
                f"(~{est_tokens} tokens in this doc, ~{total_tokens_est} total so far)",
                end="",
                file=sys.stderr,
            )
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        if total_docs:
            print(file=sys.stderr)
            print(
                f"  Recursive chunking done: {total_docs} documents → {len(all_chunks)} chunks "
                f"(~{total_tokens_est} tokens processed)",
                file=sys.stderr,
            )
        return all_chunks

    def _semantic_chunk_documents(
        self,
        documents: list,
        max_chunk_size: int = 1200,
        similarity_threshold: float = 0.8,
    ) -> list:
        """Semantic chunking using Google AI embeddings with progress."""
        chunked_docs = []
        total_tokens_est = 0
        total_tokens_from_api = 0
        request_count = 0

        doc_paragraph_counts = []
        for doc in documents:
            text = doc.page_content or ""
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            if paras:
                doc_paragraph_counts.append((doc, paras))

        for doc_idx, (doc, paragraphs) in enumerate(doc_paragraph_counts):
            current_chunk = ""
            current_embedding = None

            for para_idx, paragraph in enumerate(paragraphs):
                request_count += 1
                est_tokens = self._estimate_tokens(paragraph)
                total_tokens_est += est_tokens
                token_display = f"~{est_tokens} tokens"
                print(
                    f"\r  Semantic chunking: doc {doc_idx + 1}/{len(doc_paragraph_counts)}, "
                    f"paragraph {para_idx + 1}/{len(paragraphs)} — request #{request_count} {token_display} "
                    f"(total ~{total_tokens_est})",
                    end="",
                    file=sys.stderr,
                )
                try:
                    response = genai.embed_content(
                        model=self.embedding_model_name,
                        content=paragraph,
                    )
                    paragraph_embedding = response.get("embedding")
                    usage = response.get("usage_metadata")
                    if isinstance(usage, dict) and "total_token_count" in usage:
                        api_tokens = usage["total_token_count"]
                        total_tokens_from_api += api_tokens
                        token_display = f"{api_tokens} tokens"
                except Exception as e:
                    print(
                        f"\nError embedding paragraph for semantic chunking: {e}",
                        file=sys.stderr,
                    )
                    paragraph_embedding = None

                if not current_chunk:
                    current_chunk = paragraph
                    current_embedding = paragraph_embedding
                    continue

                combined_length = len(current_chunk) + 2 + len(paragraph)
                similarity_ok = False
                if (
                    current_embedding is not None
                    and paragraph_embedding is not None
                ):
                    similarity = self._cosine_similarity(
                        current_embedding, paragraph_embedding
                    )
                    similarity_ok = similarity >= similarity_threshold

                if (
                    similarity_ok
                    and combined_length <= max_chunk_size
                ):
                    current_chunk = current_chunk + "\n\n" + paragraph
                    current_embedding = (
                        paragraph_embedding or current_embedding
                    )
                else:
                    chunked_docs.append(
                        Document(
                            page_content=current_chunk,
                            metadata=doc.metadata,
                        )
                    )
                    current_chunk = paragraph
                    current_embedding = paragraph_embedding

            if current_chunk:
                chunked_docs.append(
                    Document(page_content=current_chunk, metadata=doc.metadata)
                )

        if request_count:
            print(file=sys.stderr)
            token_summary = (
                f"{total_tokens_from_api} tokens (API)"
                if total_tokens_from_api
                else f"~{total_tokens_est} tokens (est.)"
            )
            print(
                f"  Semantic chunking done: {request_count} embedding requests, "
                f"{len(chunked_docs)} chunks ({token_summary})",
                file=sys.stderr,
            )
        return chunked_docs

    @staticmethod
    def _cosine_similarity(v1: list, v2: list) -> float:
        """Cosine similarity between two embedding vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return dot / (n1 * n2)

    def _store_documents(self, documents: list) -> "chromadb.Collection":
        """Embed and upsert documents into the ChromaDB collection."""
        collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        for index, doc in enumerate(documents):
            doc_hash = hashlib.md5(
                doc.page_content.encode("utf-8")
            ).hexdigest()
            response = genai.embed_content(
                model=self.embedding_model_name,
                content=doc.page_content,
            )
            embedding = response["embedding"]
            collection.upsert(
                ids=[doc_hash],
                embeddings=[embedding],
                documents=[doc.page_content],
                metadatas=[
                    {
                        "source": self.source_label,
                        "file_name": doc.metadata.get("file_name", "unknown"),
                    }
                ],
            )
            print(
                f"Document {index} with hash {doc_hash} added to collection.",
                file=sys.stderr,
            )
            time.sleep(0.5)
        return collection

    def populate(self, update_data_dirs: list) -> None:
        """
        Load documents from the given directories, chunk them, and store
        in the vector DB. Follows the same logic as __main__ in WineBot.
        """
        all_raw_documents = []
        for directory in update_data_dirs:
            print(f"Loading documents from {directory}...", file=sys.stderr)
            if os.path.exists(directory):
                documents = self._load_documents(directory)
                all_raw_documents.extend(documents)
            else:
                print(
                    f"Directory {directory} does not exist.",
                    file=sys.stderr,
                )

        if not all_raw_documents:
            print("No documents found to update.", file=sys.stderr)
            return

        print(
            f"Applying '{self.chunking_mode}' chunking strategy to documents...",
            file=sys.stderr,
        )
        chunked_documents = self._chunk_documents(all_raw_documents)
        collection = self._store_documents(chunked_documents)
        print(
            f"Collection {collection.name} created/updated to {collection.count()} documents.",
            file=sys.stderr,
        )
