"""
Handles loading documents, chunking, and populating the ChromaDB vector store.
"""
import glob
import hashlib
import math
import os
import re
import sys
import time
from abc import ABC, abstractmethod

import chromadb
from google import genai
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pdf2image import convert_from_path
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

try:
    import pandas as pd
    _EXCEL_AVAILABLE = True
except ImportError:
    _EXCEL_AVAILABLE = False

CHARS_PER_TOKEN_ESTIMATE = 4
PAGE_TEXT_OCR_THRESHOLD = 50  # pages with fewer non-whitespace chars trigger OCR fallback

SLIDE_BOUNDARY_RE = re.compile(
    r'^<!-- SLIDE_BOUNDARY id=(\d+) image="([^"]*)" -->\s*$', re.MULTILINE
)

# (Windows Only) Set these if pdf2image/tesseract are not in your system PATH:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = None


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


class DocumentLoaderStrategy(ABC):
    """Loads raw Documents of a particular file format from a directory tree."""

    @abstractmethod
    def load(self, directory: str) -> list:
        """Return all Documents this strategy can read from `directory`."""


class TextDocumentLoader(DocumentLoaderStrategy):
    """Loads plain-text-like formats (md, txt, ...) via DirectoryLoader."""

    def __init__(self, extension: str):
        self.extension = extension

    def load(self, directory: str) -> list:
        loader = DirectoryLoader(
            directory, glob=f"**/*.{self.extension}", loader_cls=CustomTextLoader
        )
        return loader.load()


class PdfDocumentLoader(DocumentLoaderStrategy):
    """Loads PDFs, OCR-ing only pages whose text layer is empty/sparse."""

    def __init__(self, extract_pdf_images: bool = False):
        self.extract_pdf_images = extract_pdf_images

    def load(self, directory: str) -> list:
        documents = []
        for pdf_path in glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True):
            try:
                documents.extend(self._load_one(pdf_path))
            except Exception as e:
                print(f"Error loading PDF {pdf_path}: {e}", file=sys.stderr)
        return documents

    def _load_one(self, file_path: str) -> list:
        text_docs = PyPDFLoader(file_path, extract_images=False).load()

        if not self.extract_pdf_images:
            return text_docs

        needs_ocr = [
            i for i, d in enumerate(text_docs)
            if len((d.page_content or "").strip()) < PAGE_TEXT_OCR_THRESHOLD
        ]
        if not needs_ocr:
            return text_docs

        print(
            f"  {os.path.basename(file_path)}: {len(needs_ocr)}/{len(text_docs)} "
            f"pages need OCR fallback",
            file=sys.stderr,
        )

        if not _OCR_AVAILABLE:
            print("OCR unavailable. Please install pdf2image and pytesseract.", file=sys.stderr)
            return text_docs

        for i in needs_ocr:
            page_num = i + 1  # convert_from_path is 1-indexed
            try:
                print(f"    Running OCR on page {page_num}...", file=sys.stderr)
                images = convert_from_path(
                    file_path,
                    first_page=page_num,
                    last_page=page_num,
                    poppler_path=POPPLER_PATH,
                )
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    if ocr_text.strip():
                        text_docs[i] = Document(
                            page_content=ocr_text,
                            metadata=text_docs[i].metadata,
                        )
            except Exception as e:
                print(f"  OCR error on page {page_num}: {e}", file=sys.stderr)

        return text_docs


class ExcelDocumentLoader(DocumentLoaderStrategy):
    """Loads Excel workbooks, emitting one Document per non-empty worksheet."""

    def __init__(self, extension: str = "xlsx"):
        self.extension = extension

    def load(self, directory: str) -> list:
        if not _EXCEL_AVAILABLE:
            print(
                "Excel support unavailable. Please install pandas and openpyxl (xlrd for .xls).",
                file=sys.stderr,
            )
            return []
        documents = []
        for path in glob.glob(os.path.join(directory, "**", f"*.{self.extension}"), recursive=True):
            try:
                documents.extend(self._load_one(path))
            except Exception as e:
                print(f"Error loading Excel {path}: {e}", file=sys.stderr)
        return documents

    @staticmethod
    def _load_one(file_path: str) -> list:
        # sheet_name=None returns a dict of {sheet_name: DataFrame} for every sheet.
        sheets = pd.read_excel(file_path, sheet_name=None)
        file_name = os.path.basename(file_path)
        documents = []
        for sheet_name, frame in sheets.items():
            if frame.empty:
                continue
            documents.append(
                Document(
                    page_content=frame.to_csv(index=False),
                    metadata={
                        "file_name": file_name,
                        "source": file_path,
                        "sheet": sheet_name,
                    },
                )
            )
        return documents


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
        extract_pdf_images: bool = False,
    ):
        """
        Args:
            api_key: Google API key for embeddings.
            embedding_model_name: Model name for embed_content (e.g. gemini-embedding-001).
            collection_name: ChromaDB collection name.
            db_path: Path to the ChromaDB persistence directory.
            chunking_mode: "recursive" or "semantic".
        """
        self.genai_client = genai.Client(api_key=api_key)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.db_path = db_path
        self.chunking_mode = chunking_mode
        self.source_label = source_label
        self.client = chromadb.PersistentClient(path=db_path)
        self.extract_pdf_images = extract_pdf_images

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(0, len(text) // CHARS_PER_TOKEN_ESTIMATE)

    def _make_loader(self, extension: str) -> DocumentLoaderStrategy:
        """Pick the loader strategy for a given file extension."""
        if extension == "pdf":
            return PdfDocumentLoader(self.extract_pdf_images)
        if extension in ("xlsx", "xls"):
            return ExcelDocumentLoader(extension)
        return TextDocumentLoader(extension)

    @staticmethod
    def _basename(doc: Document) -> str:
        source_path = doc.metadata.get("source", doc.metadata.get("file_name", ""))
        return os.path.splitext(os.path.basename(source_path))[0]

    def _load_documents(self, directory: str, load_types=None) -> list:
        """Load raw documents from a directory. Falls back to OCR if text formats yield nothing."""
        documents = []
        if not os.path.exists(directory):
            return documents

        if load_types is None:
            load_types = ["md", "txt"]
        elif isinstance(load_types, str):
            load_types = [load_types]

        loaded_basenames = set()

        def collect(extension: str) -> None:
            try:
                loaded_docs = self._make_loader(extension).load(directory)
            except Exception as e:
                print(f"Error loading {extension} files: {e}", file=sys.stderr)
                return
            # Skip files already loaded by an earlier format, but keep every
            # doc from a new file (PDFs yield one Document per page).
            seen_before = set(loaded_basenames)
            for doc in loaded_docs:
                base_name = self._basename(doc)
                if base_name in seen_before:
                    continue
                documents.append(doc)
                loaded_basenames.add(base_name)

        for ext in load_types:
            collect(ext)

        # If the intended text formats found nothing, fall back to PDFs (with OCR).
        if not documents and "pdf" not in load_types:
            print(f"No {load_types} documents found in {directory}. Defaulting to OCR fallback.", file=sys.stderr)
            collect("pdf")

        return documents

    def _chunk_documents(self, documents: list) -> list:
        """Apply the configured chunking strategy."""
        if self.chunking_mode == "semantic":
            return self._semantic_chunk_documents(documents)
        if self.chunking_mode == "slide":
            return self._slide_chunk_documents(documents)
        return self._recursive_chunk_documents(documents)

    def _slide_chunk_documents(
        self,
        documents: list,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> list:
        """Split slide-pipeline markdown on SLIDE_BOUNDARY markers, one chunk per
        slide (further split only if a slide's own text exceeds chunk_size).
        Every resulting chunk carries `image_path`/`slide_number` metadata so the
        original diagram can be re-attached at answer time. Documents with no
        SLIDE_BOUNDARY markers (e.g. hand-curated notes mixed into the same data
        dir) fall back to ordinary recursive chunking.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_chunks = []
        for doc in documents:
            content = doc.page_content or ""
            matches = list(SLIDE_BOUNDARY_RE.finditer(content))
            if not matches:
                all_chunks.extend(self._recursive_chunk_documents([doc], chunk_size, chunk_overlap))
                continue

            for match_index, match in enumerate(matches):
                slide_number = int(match.group(1))
                image_path = match.group(2)
                section_start = match.end()
                section_end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(content)
                section_text = content[section_start:section_end].strip()
                if not section_text:
                    continue

                slide_metadata = {**doc.metadata, "image_path": image_path, "slide_number": slide_number}

                if len(section_text) <= chunk_size:
                    all_chunks.append(
                        Document(page_content=section_text, metadata={**slide_metadata, "_chunk_index": 0})
                    )
                else:
                    sub_chunks = text_splitter.split_documents(
                        [Document(page_content=section_text, metadata=slide_metadata)]
                    )
                    for sub_index, sub_chunk in enumerate(sub_chunks):
                        sub_chunk.metadata["_chunk_index"] = sub_index
                    all_chunks.extend(sub_chunks)

        print(
            f"  Slide chunking done: {len(documents)} documents -> {len(all_chunks)} chunks",
            file=sys.stderr,
        )
        return all_chunks

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
        request_count = 0

        doc_paragraphs = [
            (doc, [p.strip() for p in (doc.page_content or "").split("\n\n") if p.strip()])
            for doc in documents
        ]
        doc_paragraphs = [(doc, paras) for doc, paras in doc_paragraphs if paras]

        for doc_idx, (doc, paragraphs) in enumerate(doc_paragraphs):
            current_chunk = ""
            current_embedding = None

            for para_idx, paragraph in enumerate(paragraphs):
                request_count += 1
                est_tokens = self._estimate_tokens(paragraph)
                total_tokens_est += est_tokens
                print(
                    f"\r  Semantic chunking: doc {doc_idx + 1}/{len(doc_paragraphs)}, "
                    f"paragraph {para_idx + 1}/{len(paragraphs)} — request #{request_count} "
                    f"~{est_tokens} tokens (total ~{total_tokens_est})",
                    end="",
                    file=sys.stderr,
                )
                try:
                    response = self.genai_client.models.embed_content(
                        model=self.embedding_model_name,
                        contents=paragraph,
                    )
                    paragraph_embedding = response.embeddings[0].values
                except Exception as e:
                    print(f"\nError embedding paragraph for semantic chunking: {e}", file=sys.stderr)
                    paragraph_embedding = None

                if not current_chunk:
                    current_chunk = paragraph
                    current_embedding = paragraph_embedding
                    continue

                combined_length = len(current_chunk) + 2 + len(paragraph)
                similarity_ok = (
                    current_embedding is not None
                    and paragraph_embedding is not None
                    and self._cosine_similarity(current_embedding, paragraph_embedding) >= similarity_threshold
                )

                if similarity_ok and combined_length <= max_chunk_size:
                    current_chunk = current_chunk + "\n\n" + paragraph
                    current_embedding = paragraph_embedding or current_embedding
                else:
                    chunked_docs.append(Document(page_content=current_chunk, metadata=doc.metadata))
                    current_chunk = paragraph
                    current_embedding = paragraph_embedding

            if current_chunk:
                chunked_docs.append(Document(page_content=current_chunk, metadata=doc.metadata))

        if request_count:
            print(file=sys.stderr)
            print(
                f"  Semantic chunking done: {request_count} embedding requests, "
                f"{len(chunked_docs)} chunks (~{total_tokens_est} tokens est.)",
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
        collection = self.client.get_or_create_collection(name=self.collection_name)
        for index, doc in enumerate(documents):
            file_name = doc.metadata.get("file_name", "unknown")
            image_path = doc.metadata.get("image_path")

            metadata = {"source": self.source_label, "file_name": file_name}
            if image_path:
                metadata["content_type"] = "image"
                metadata["image_path"] = image_path
                if doc.metadata.get("slide_number") is not None:
                    metadata["slide_number"] = int(doc.metadata["slide_number"])
                # Stable per-slide id: Gemini's slide descriptions aren't
                # deterministic between runs, so a content-hash id would
                # accumulate duplicate/stale rows every time the
                # markdown gets regenerated while tuning the diagram prompt.
                chunk_index = doc.metadata.get("_chunk_index", 0)
                doc_id = f"{file_name}:slide{metadata['slide_number']}:{chunk_index}"
            else:
                metadata["content_type"] = "text"
                doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

            response = self.genai_client.models.embed_content(
                model=self.embedding_model_name,
                contents=doc.page_content,
            )
            embedding = response.embeddings[0].values
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc.page_content],
                metadatas=[metadata],
            )
            print(f"Document {index} with id {doc_id} added to collection.", file=sys.stderr)
            time.sleep(0.5)
        return collection

    def populate(self, update_data_dirs: list, load_types: list = None) -> None:
        """Load documents from the given directories, chunk them, and store in the vector DB."""
        all_raw_documents = []
        for directory in update_data_dirs:
            print(f"Loading documents from {directory}...", file=sys.stderr)
            if os.path.exists(directory):
                documents = self._load_documents(directory, load_types=load_types)
                all_raw_documents.extend(documents)
            else:
                print(f"Directory {directory} does not exist.", file=sys.stderr)

        if not all_raw_documents:
            print("No documents found to update.", file=sys.stderr)
            return

        print(f"Applying '{self.chunking_mode}' chunking strategy to documents...", file=sys.stderr)
        chunked_documents = self._chunk_documents(all_raw_documents)
        collection = self._store_documents(chunked_documents)
        print(
            f"Collection {collection.name} created/updated to {collection.count()} documents.",
            file=sys.stderr,
        )
