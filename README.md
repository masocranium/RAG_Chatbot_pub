# Generic RAG Chatbot Framework

A configurable **RAG (Retrieval-Augmented Generation)** framework for building chatbots over your own documents, using Gemini and a local Chroma vector store. No documents are included, so you bring your own.

## Features

- **Reusable core:** [rag_core.py](rag_core.py) handles retrieval and response generation. You supply a knowledge-base config and a prompt config.
- **Document ingestion:** [vector_db_populator.py](vector_db_populator.py) loads Markdown, text, PDF (with OCR fallback) and Excel files.
- **Chunking modes:** `recursive`, `semantic`, and a diagram-aware `slide` mode that re-attaches the original slide image when answering.
- **Conversation memory:** [conversation_memory.py](conversation_memory.py) offers no memory, a buffer, or a rolling summary.
- **PDF to Markdown:** [pdf_to_markdown.py](pdf_to_markdown.py) converts presentation PDFs using Gemini vision or Tesseract.
- **Anki deck generation:** [anki_core.py](anki_core.py) turns Markdown notes into a tab-separated file and a native `.apkg` deck.
- **Web interface:** [web_app.py](web_app.py) is a small Flask chat UI that accepts pasted images.

## Setup

Developed on Python 3.12.

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=<YOUR GOOGLE API KEY>" > .env
```

`data/` and `db/` are ignored by git, so your documents and vector store stay local.

## Building the vector DB

1. Put your documents in `data/example/documents/`.
2. In [models/ExampleBot.py](models/ExampleBot.py), set `UPDATE_CHROMA_DB = True`.
3. Run the preset:

   ```bash
   python models/ExampleBot.py
   ```

4. Set `UPDATE_CHROMA_DB` back to `False` once the build finishes. The database is saved in `db/vector_dbs/example_db/`.

## Running the chatbot

From the terminal:

```bash
python models/ExampleBot.py
```

In the browser, at http://127.0.0.1:5000:

```bash
python web_app.py
```

## Generating an Anki deck

1. Put Markdown notes in `data/example/md_notes/`. Each file becomes one batch of cards.
2. Run the preset:

   ```bash
   python models/ExampleAnkiBot.py
   ```

The deck is written to `data/example/anki_output/`. To calibrate the style of the cards, add sample questions to `data/example/example_questions/` and set `INGEST_EXAMPLE_QUESTIONS = True`.

## Converting PDFs to Markdown

```bash
python pdf_to_markdown.py <input_dir> <output_dir>
```

## Creating your own chatbot

Copy [models/ExampleBot.py](models/ExampleBot.py), then change its data directories, collection name and `PromptConfig`.

## Credits

masocranium

alberto-escobar
