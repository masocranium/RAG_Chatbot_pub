"""Example Anki deck generator preset built on top of the Anki core.

Put markdown files in data/example/md_notes/ (one file becomes one batch of cards).
Optionally add sample questions as .md/.txt files in data/example/example_questions/
and set INGEST_EXAMPLE_QUESTIONS = True to calibrate the style of generated cards.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow importing project modules from the parent directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from anki_core import AnkiCardBot, AnkiConfig
from rag_core import KnowledgeBaseConfig, PromptConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FLDR = os.path.join(PROJECT_ROOT, "data", "example")
MD_NOTES_DIR = os.path.join(DATA_FLDR, "md_notes")
EXAMPLE_QUESTIONS_DIR = os.path.join(DATA_FLDR, "example_questions")
OUTPUT_PATH = os.path.join(DATA_FLDR, "anki_output", "example_cards.txt")
DB_DESTINATION_PATH = os.path.join(PROJECT_ROOT, "db", "vector_dbs", "example_anki_db")
COLLECTION_NAME = "example_questions"

DEFAULT_PROMPT = PromptConfig(
    assistant_role="an expert tutor who writes sharp, exam-style flashcards",
    source_description="the supplied notes",
)


class ExampleAnkiBot(AnkiCardBot):
    """A minimal preset showing how to configure the Anki deck generator."""

    def __init__(self, llm_model_name: str, embedding_model_name: str, api_key: str):
        knowledge_base = KnowledgeBaseConfig(
            collection_name=COLLECTION_NAME,
            db_path=DB_DESTINATION_PATH,
            data_dirs=[EXAMPLE_QUESTIONS_DIR],
            source_label="Example Questions",
            fallback_search_query="example exam question",
            retrieval_k=5,
        )
        anki_config = AnkiConfig(
            deck_name="Example",
            output_path=OUTPUT_PATH,
            source_md_dir=MD_NOTES_DIR,
            base_tags=["Example"],
        )
        super().__init__(
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            api_key=api_key,
            knowledge_base=knowledge_base,
            anki_config=anki_config,
            prompt_config=DEFAULT_PROMPT,
        )


if __name__ == "__main__":
    load_dotenv()

    INGEST_EXAMPLE_QUESTIONS = False  # (Re)build the example-questions collection.

    anki_bot = ExampleAnkiBot(
        llm_model_name="gemini-3.1-flash-lite",
        embedding_model_name="gemini-embedding-001",
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    if INGEST_EXAMPLE_QUESTIONS:
        if any(Path(EXAMPLE_QUESTIONS_DIR).glob("**/*")):
            os.makedirs(DB_DESTINATION_PATH, exist_ok=True)
            anki_bot.update_vector_db([EXAMPLE_QUESTIONS_DIR], load_types=["md", "txt"])
        else:
            print(f"No example questions found in {EXAMPLE_QUESTIONS_DIR}; skipping ingestion.")

    summary = anki_bot.generate_deck()
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
