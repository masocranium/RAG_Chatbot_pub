# Generic RAG Chatbot Framework

This repository provides a configurable **RAG (Retrieval-Augmented Generation) core** in [rag_core.py](rag_core.py) that makes it easy to build domain-specific chatbots backed by your own knowledge bases. The framework handles vector-store population, embedding management, and response generation through a reusable, customizable API.

## Example: Wine Science Chatbot

As a concrete example, this project includes `WineBot.py`, a wine science knowledge chatbot built on top of the RAG core. It demonstrates how to configure the framework for a specific domain by defining custom knowledge-base sources, embedding models, and prompts.

## Setups
1. `pip install -r requirements.txt`
2. `echo "GOOGLE_API_KEY=<YOUR GOOGLE API KEY>" > .env`
3. `python WineBot.py`

To adapt this project for a different RAG domain, create a new preset on top of [rag_core.py](rag_core.py) with a different knowledge-base config and prompt config.

any issues running? read the errors or ask AI.

## Building the vector DB

Follow these steps to build (populate) the Chroma vector database from the source files in `wine_data`.

- Ensure your API key is set in `.env`:

	```powershell
	echo "GOOGLE_API_KEY=<YOUR GOOGLE API KEY>" > .env
	```

- Install dependencies (on Windows the full `requirements.txt` may try to build `uvloop`; if that fails, see the troubleshooting note below):

	```bash
	pip install -r requirements.txt
	```

- Option A — helper script (recommended): create a file named `build_db.py` with the contents below, then run it.

	File: `build_db.py`

	```python
	from dotenv import load_dotenv
	import os

	load_dotenv()

	from WineBot import WineChatbot

	bot = WineChatbot(
			llm_model_name="gemini-2.5-flash",
			embedding_model_name="gemini-embedding-001",
			api_key=os.environ.get("GOOGLE_API_KEY"),
			collection_name="fnh330_wine_course_gemini",
			db_path="fnh330_db/",
			chunking_mode="recursive",
	)

	if not os.path.exists(bot.knowledge_base.db_path):
			os.makedirs(bot.knowledge_base.db_path)

	bot.update_vector_db(bot.knowledge_base.data_dirs)
	```

	Run:

	```powershell
	python build_db.py
	```

- Option B — in-file toggle: open `WineBot.py`, set `UPDATE_CHROMA_DB = True` near the top, then run:

	```powershell
	python WineBot.py
	```

- Verify: after the script completes, the Chroma DB will be persisted under `fnh330_db/` (or the `db_path` you configured).

Troubleshooting
- If `pip install -r requirements.txt` fails on Windows due to `uvloop`, install the core runtime packages first and skip `uvloop`:

	```powershell
	pip install chromadb google-generativeai langchain-core langchain-text-splitters python-dotenv pypdf
	```

	Then run `python build_db.py` as above.


## Credits
masocranium

alberto-escobar
