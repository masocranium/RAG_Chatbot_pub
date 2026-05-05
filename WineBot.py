"""Wine-specific preset built on top of the generic RAG core."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from rag_core import GenericRAGChatbot, KnowledgeBaseConfig, PromptConfig


DB_DESTINATION_PATH = "fnh330_db/"
USE_SEMANTIC_CHUNKING = False

DEFAULT_WINE_DATA_DIRS = [
    os.path.join("wine_data", "Lectures"),
    os.path.join("wine_data", "Notes"),
    os.path.join("wine_data", "video_transcripts"),
]

DEFAULT_WINE_PROMPT = PromptConfig(
    assistant_role="a helpful teaching assistant",
    source_description="the FNH 330 Wine Science course materials",
    answer_style=(
        "Your answer should be concise. Provide the explanation first, then the answer on a new line clearly marked as the answer."
    ),
    citation_instruction="Cite any specific lecture notes or files where possible.",
    image_instruction=(
        "Please carefully analyze the attached image of the wine label to determine its classification, style, and processing considerations."
    ),
    empty_prompt_fallback="What can you tell me about the wine in this image?",
)


class WineChatbot(GenericRAGChatbot):
    """Compatibility wrapper for the wine course preset."""

    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        api_key: str,
        collection_name: str,
        db_path: str,
        chunking_mode: str = "recursive",
    ):
        knowledge_base = KnowledgeBaseConfig(
            collection_name=collection_name,
            db_path=db_path,
            data_dirs=list(DEFAULT_WINE_DATA_DIRS),
            source_label="FNH330 Course Material",
            chunking_mode=chunking_mode,
            fallback_search_query="wine label formal classification",
            retrieval_k=7,
        )
        super().__init__(
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            api_key=api_key,
            knowledge_base=knowledge_base,
            prompt_config=DEFAULT_WINE_PROMPT,
        )


def get_multiline_input(prompt_message):
    """
    Read multiple lines of input until the user types DONE.
    """
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

    if not lines:
        return ""

    return "\n".join(lines)


if __name__ == "__main__":
    load_dotenv()

    UPDATE_CHROMA_DB = False
    RAG_ONLY_MODE = False

    wine_chatbot = WineChatbot(
        llm_model_name="gemini-2.5-flash",
        embedding_model_name="gemini-embedding-001",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        collection_name="fnh330_wine_course_gemini",
        db_path=DB_DESTINATION_PATH,
        chunking_mode="semantic" if USE_SEMANTIC_CHUNKING else "recursive",
    )

    print("Updating database with wine course materials...")

    if UPDATE_CHROMA_DB:
        if not os.path.exists(DB_DESTINATION_PATH):
            os.makedirs(DB_DESTINATION_PATH)
        wine_chatbot.update_vector_db(wine_chatbot.knowledge_base.data_dirs)

    print("Ready to go! Type 'quit' to exit.")
    while True:
        if RAG_ONLY_MODE:
            prompt = get_multiline_input(
                "\nPlease enter your question about wine science (RAG mode - documents only):"
            )
            if prompt.strip().lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            relevant_docs = wine_chatbot.retrieve_docs_only(prompt, k=7)

            documents = relevant_docs["documents"][0]
            metadatas = relevant_docs["metadatas"][0]

            print(f"\n--- Found {len(documents)} Relevant Documents ---")

            for i, doc in enumerate(documents):
                print(f"\n{'='*50}")
                print(f"DOCUMENT {i+1}")
                if i < len(metadatas):
                    print(f"Source file: {metadatas[i].get('file_name', 'Unknown')}")
                print(f"{'-'*50}")
                print(doc.strip())
                print(f"{'='*50}\n")
        else:
            prompt = get_multiline_input("\nPlease enter your question about wine science:")
            if prompt.strip().lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            print(wine_chatbot.run(prompt))