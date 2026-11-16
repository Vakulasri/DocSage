# DocSage: RAG Document Q&A with Evaluation

Ask questions about SEC 10-K filings and get answers with a page-level citation for every fact. If the answer isn't in the documents, DocSage says so instead of guessing. It also includes an evaluation pipeline that scores retrieval accuracy, answer accuracy and faithfulness (whether it made anything up).

**Built by Vakula Bhima** · [GitHub](https://github.com/Vakulasri/DocSage)

**Stack:** Python, ChromaDB, Groq / Gemini APIs, Streamlit

## Results

Tested on 19 questions over 435 pages of 10-K filings (Apple, Microsoft and NVIDIA, fiscal 2026; 2,001 indexed chunks):

| Metric | Score | What it measures |
|---|---|---|
| Retrieval accuracy | 100% | The right document was found (15 questions with an expected source) |
| Answer accuracy | 100% | The answer contained the expected facts |
| Faithfulness | 89% | An LLM judge found no unsupported claims |
| Avg latency | 16.2s | Time per answer, including multi-step questions |

Both faithfulness misses were correct "I couldn't find that" answers that the first judge prompt scored as wrong. The judge now checks refusals separately; rerun `python eval.py` for the current number.

## How it works

1. **Indexing:** each PDF is split into overlapping chunks, tagged with its document name, and stored in ChromaDB.
2. **Hybrid search:** every question is searched two ways, by meaning (vector search) and by exact words (BM25 keyword search). The two rankings are merged with reciprocal rank fusion, so exact figures aren't missed.
3. **Company-aware retrieval:** if a question names a company, only that company's filing is searched. "Which company" and comparison questions search each filing separately so every company gets its own passages.
4. **Multi-query retrieval:** questions that need several facts, like revenue per employee, are split into sub-queries ("Apple total net sales", "Apple number of employees") and each is searched separately.
5. **Answering:** the LLM answers only from the retrieved passages, cites each fact as [1], [2], and shows its math for calculations.

## App features

- Answers stream in word by word, with citation badges and source cards (file and page)
- "Searched for" chips showing the sub-queries behind each answer
- Upload and index a new PDF from the sidebar
- Export the conversation as Markdown or PDF
- Evaluation tab with score cards and per-question results

## Setup on Windows (PowerShell)

1. Open this folder in VS Code and open a terminal (Ctrl + `).
2. Create and activate a virtual environment:
   ```
   py -m venv venv
   venv\Scripts\activate
   ```
   If you get a "running scripts is disabled" error, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```
4. Get a free API key from https://console.groq.com/keys (or Gemini: https://aistudio.google.com/apikey).
5. Copy `.env.example` to a new file named `.env` and fill it in:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=your-key
   GROQ_MODEL=openai/gpt-oss-120b
   TOP_K=8
   ```
   To use Gemini instead, set `LLM_PROVIDER=gemini`, `GEMINI_API_KEY` and `GEMINI_MODEL`.

## Run it

1. Put PDFs in the `data` folder, named by company (for example `Apple-10k.pdf`) so questions can target them. Use text-based PDFs: scanned image PDFs can't be read.
2. Index them (the first run downloads a small local embedding model):
   ```
   python ingest.py
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```

## Evaluate it

1. Edit `eval_set.json`. Each question has an `expected_source` (file name, or `null` for comparison and trick questions) and `expected_keywords` (facts a correct answer must contain). Include a few questions the documents can't answer.
2. Run:
   ```
   python eval.py
   ```
   Add `EVAL_DELAY=3` to `.env` to speed it up on Groq.
3. Results are saved to `eval_results.json` and shown in the app's Evaluation tab. Rerun after changing the code, PDFs, questions or model.

## Deploy it (Streamlit Community Cloud)

1. Make sure the PDFs and `eval_results.json` are in the repo. Remove `eval_results.json` from `.gitignore` if it's listed, so the Evaluation tab works online.
2. Push to GitHub:
   ```
   git init
   git add .
   git commit -m "DocSage RAG app"
   git branch -M main
   git remote add origin https://github.com/Vakulasri/DocSage.git
   git push -u origin main
   ```
   Check on GitHub that `.env` and `venv` did **not** upload.
3. Go to https://share.streamlit.io, click **Create app**, and pick the repo, the `main` branch and `app.py`.
4. Under **Advanced settings**, choose Python **3.11** and paste into **Secrets**:
   ```
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your-real-key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```
5. Click **Deploy**. The first load takes a few minutes while it builds the index.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `rag.py` | Retrieval (hybrid search, multi-query) and answering |
| `ingest.py` | PDF loading, chunking and indexing |
| `eval.py` | Evaluation pipeline with LLM-as-judge |
| `eval_set.json` | Test questions |
| `.streamlit/config.toml` | App theme |

## Troubleshooting

- **No such file `requirements.txt` or `ingest.py`:** you're one folder too high. `cd` into the inner `rag-docqa` folder.
- **Model not found:** list the models your key can use and set `GROQ_MODEL` or `GEMINI_MODEL` to one of them.
- **503 or 429 errors:** the provider is busy or you hit a free-tier limit. The app retries automatically; wait a minute or switch providers.
- **A PDF shows "not indexed":** it's a scanned image with no text. Download a text-based copy.
- **"No documents indexed":** run `python ingest.py` after adding PDFs.
