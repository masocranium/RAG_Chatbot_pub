"""Example chatbot preset built on top of the generic RAG core.

Copy this file and change the paths and prompts to build a chatbot for your own
documents. Put source files in data/example/documents/ before building the DB.
"""
from __future__ import annotations

import os
import sys

# Allow importing project modules from the parent directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from pdf_to_markdown import preprocess_pdfs_to_markdown
from rag_core import GenericRAGChatbot, KnowledgeBaseConfig, PromptConfig, build_memory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLLECTION_NAME = "example_collection"
DB_DESTINATION_PATH = os.path.join(PROJECT_ROOT, "db", "vector_dbs", "example_db")

DATA_FLDR = os.path.join(PROJECT_ROOT, "data", "example")
DOCUMENTS_DIR = os.path.join(DATA_FLDR, "documents")
# Used only by "slide" mode, which converts presentation PDFs before indexing them.
SRC_PDF_DIR = os.path.join(DATA_FLDR, "src_pdfs")
GENERATED_MD_DIR = os.path.join(DATA_FLDR, "md_generated")
IMAGES_DIR = os.path.join(DATA_FLDR, "slide_images")

# "recursive" -> fixed-size text chunks.
# "semantic"  -> split on embedding similarity.
# "slide"     -> one chunk per slide, re-attaching the slide image when answering.
CHUNKING_MODE = "recursive"

DEFAULT_PROMPT = PromptConfig(
    assistant_role="a helpful assistant",
    source_description="the supplied documents",
    answer_style="Be concise. Give the explanation first, then the answer on a new line.",
    citation_instruction="Cite specific file names where possible.",
)


class ExampleChatbot(GenericRAGChatbot):
    """A minimal preset showing how to configure the RAG core."""

    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        api_key: str,
        collection_name: str,
        db_path: str,
        chunking_mode: str = CHUNKING_MODE,
        enable_memory: bool = False,
        max_history_turns: int | None = None,
        memory_mode: str = "buffer",
        summarization_model: str | None = None,
    ):
        knowledge_base = KnowledgeBaseConfig(
            collection_name=collection_name,
            db_path=db_path,
            data_dirs=[GENERATED_MD_DIR] if chunking_mode == "slide" else [DOCUMENTS_DIR],
            source_label="Example Documents",
            chunking_mode=chunking_mode,
            fallback_search_query="overview of key concepts",
            retrieval_k=7,
            images_base_dir=PROJECT_ROOT,
        )
        memory = build_memory(
            api_key,
            llm_model_name,
            enable_memory=enable_memory,
            memory_mode=memory_mode,
            max_history_turns=max_history_turns,
            summarization_model=summarization_model,
        )
        super().__init__(
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            api_key=api_key,
            knowledge_base=knowledge_base,
            prompt_config=DEFAULT_PROMPT,
            memory=memory,
        )


def get_multiline_input(prompt_message):
    """Read multiple lines of input until the user types DONE."""
    print(prompt_message)
    print("(Paste your text. Type 'DONE' on a new line and press Enter to submit.)\n")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "DONE":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


if __name__ == "__main__":
    load_dotenv()

    UPDATE_CHROMA_DB = False  # Set to True to (re)build the vector DB on startup.
    RAG_ONLY_MODE = False     # Set to True to print retrieved chunks without calling the LLM.

    api_key = os.environ.get("GOOGLE_API_KEY")

    chatbot = ExampleChatbot(
        llm_model_name="gemini-3.1-flash-lite",
        embedding_model_name="gemini-embedding-001",
        api_key=api_key,
        collection_name=COLLECTION_NAME,
        db_path=DB_DESTINATION_PATH,
    )

    if UPDATE_CHROMA_DB:
        os.makedirs(DB_DESTINATION_PATH, exist_ok=True)
        if CHUNKING_MODE == "slide":
            print("Converting PDFs to markdown...")
            preprocess_pdfs_to_markdown(
                SRC_PDF_DIR,
                GENERATED_MD_DIR,
                api_key,
                only_new_files=True,
                images_dir=IMAGES_DIR,
            )
            load_types = ["md", "txt"]
        else:
            load_types = ["pdf", "md", "txt"]
        print("Updating the vector DB...")
        chatbot.update_vector_db(chatbot.knowledge_base.data_dirs, load_types=load_types)

    print("Ready to go! Type 'quit' to exit.")
    while True:
        prompt = get_multiline_input("\nPlease enter your question:")
        if prompt.strip().lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if RAG_ONLY_MODE:
            relevant_docs = chatbot.retrieve_docs_only(prompt)
            documents = relevant_docs["documents"][0]
            metadatas = relevant_docs["metadatas"][0]
            print(f"\n--- Found {len(documents)} Relevant Documents ---")
            for i, doc in enumerate(documents):
                print(f"\n{'=' * 50}\nDOCUMENT {i + 1}")
                if i < len(metadatas):
                    print(f"Source file: {metadatas[i].get('file_name', 'Unknown')}")
                print(f"{'-' * 50}\n{doc.strip()}\n{'=' * 50}\n")
        else:
            print(chatbot.run(prompt))
