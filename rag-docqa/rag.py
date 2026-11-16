"""Core RAG logic: retrieve relevant chunks from Chroma, then ask the LLM (Gemini or Groq) to answer with citations."""
import math
import os
import re
from collections import Counter
import sys
import time

# Streamlit Cloud's Linux image has an old SQLite that Chroma rejects; swap in a newer one there.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_setting(name: str, default=None):
    """Read a setting from .env locally, or from Streamlit secrets when deployed."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


DB_DIR = "chroma_db"
COLLECTION = "docs"
CHAT_MODEL = get_setting("GEMINI_MODEL", "gemini-flash-latest")
TOP_K = int(get_setting("TOP_K", "5"))

PROVIDER = (get_setting("LLM_PROVIDER", "gemini") or "gemini").lower()
GROQ_MODEL = get_setting("GROQ_MODEL", "llama-3.3-70b-versatile")
if PROVIDER == "groq":
    CHAT_MODEL = GROQ_MODEL

_llm = None


def get_collection():
    """Open (or create) the local vector database.
    Chroma's built-in embedding model runs locally, so embeddings are free."""
    db = chromadb.PersistentClient(path=DB_DIR)
    return db.get_or_create_collection(COLLECTION)


def get_llm():
    """Create the client for whichever provider is set in .env (LLM_PROVIDER=gemini or groq)."""
    global _llm
    if _llm is None:
        if PROVIDER == "groq":
            from groq import Groq
            key = get_setting("GROQ_API_KEY")
            if not key or key.startswith("paste-"):
                raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
            _llm = Groq(api_key=key)
        else:
            key = get_setting("GEMINI_API_KEY")
            if not key or key.startswith("paste-"):
                raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env (local) or to Secrets (Streamlit Cloud).")
            _llm = genai.Client(api_key=key)
    return _llm


def _call_model(prompt: str) -> str:
    if PROVIDER == "groq":
        resp = get_llm().chat.completions.create(
            model=CHAT_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}]
        )
        return (resp.choices[0].message.content or "").strip()
    response = get_llm().models.generate_content(model=CHAT_MODEL, contents=prompt)
    return (response.text or "").strip()


def generate(prompt: str) -> str:
    """Call the model, retrying a few times when the server is busy (503) or rate-limited (429)."""
    for attempt in range(4):
        try:
            return _call_model(prompt)
        except Exception as e:
            if attempt < 3 and ("503" in str(e) or "429" in str(e)):
                time.sleep(5 * (attempt + 1))
            else:
                raise


def list_sources() -> list[str]:
    """Names of all indexed documents."""
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def mentioned_sources(question: str) -> list[str]:
    """Documents whose name (e.g. 'apple' in apple-10k.pdf) appears in the question."""
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    hits = []
    for src in list_sources():
        name_words = re.findall(r"[a-z]+", os.path.splitext(src)[0].lower())
        if any(w in words for w in name_words if len(w) > 2 and w not in {"pdf", "report", "annual"}):
            hits.append(src)
    return hits


# ---------- Keyword search (BM25) ----------
# Meaning-based search alone often misses exact facts in long reports, so we also score chunks
# by the words they share with the question, then merge both rankings (hybrid search).

STOPWORDS = set("""a an the of in on at to for by with from and or is are was were be been do does did
have has had how many much what which who whom when where why this that these those it its as
about into than then there their they them we our you your i me my last year years recent most
company companies tell give show list please""".split())

# Different reports use different words for the same thing (e.g. some say "net sales" instead of "revenue").
SYNONYMS = {
    "revenue": ["sales", "net", "revenues", "total"],
    "revenues": ["sales", "net", "revenue", "total"],
    "sales": ["revenue", "revenues"],
    "employees": ["employee", "headcount", "employed", "workforce", "full", "people"],
    "headcount": ["employees", "employed"],
    "profit": ["income", "earnings"],
    "earnings": ["income", "profit"],
    "customers": ["customer", "direct", "accounted", "concentration"],
    "customer": ["customers", "direct", "accounted", "concentration"],
    "competitors": ["competition", "compete", "competitive"],
    "r&d": ["research", "development"],
}

_corpus = {"count": None}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def _load_corpus() -> dict:
    """Load every chunk once (and again only if the index changes)."""
    col = get_collection()
    n = col.count()
    if _corpus["count"] != n:
        data = col.get(include=["documents", "metadatas"])
        toks = [_tokens(d) for d in data["documents"]]
        df = Counter()
        for t in toks:
            df.update(set(t))
        lens = [len(t) for t in toks]
        _corpus.update(count=n, ids=data["ids"], docs=data["documents"], metas=data["metadatas"],
                       tf=[Counter(t) for t in toks], lens=lens, df=df, avg=sum(lens) / max(n, 1))
    return _corpus


def keyword_search(question: str, sources: list[str] | None, k: int) -> list[str]:
    """Return chunk ids ranked by BM25 keyword score."""
    c = _load_corpus()
    name_words = {w for src in (sources or []) for w in re.findall(r"[a-z]+", src.lower())}
    q_words = set(_tokens(question)) - name_words - COMPARE_WORDS
    for w in list(q_words):
        q_words.update(SYNONYMS.get(w, []))
    N = len(c["ids"])
    q_lower = question.lower()
    wants_numbers = any(w in q_lower for w in ("how many", "how much", "revenue", "sales", "income", "profit",
                                                "employees", "compare", "total", "margin", "growth"))
    scored = []
    for i, (tf, length, meta) in enumerate(zip(c["tf"], c["lens"], c["metas"])):
        if sources and meta["source"] not in sources:
            continue
        score = 0.0
        for w in q_words:
            f = tf.get(w, 0)
            if f:
                idf = math.log(1 + (N - c["df"][w] + 0.5) / (c["df"][w] + 0.5))
                score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * length / c["avg"]))
        if score > 0 and wants_numbers:
            # Favor passages that state figures ($391.0 billion, 36,000, 15%) for "how much / how many" questions.
            figures = len(re.findall(r"\$\s?[\d,.]+|\d[\d,]*\.?\d*\s?(?:billion|million|%|percent)", c["docs"][i], re.I))
            score *= 1 + min(figures, 6) * 0.15
        if score > 0:
            scored.append((score, c["ids"][i]))
    scored.sort(reverse=True)
    return [cid for _, cid in scored[:k]]


COMPARE_WORDS = {"compare", "comparison", "companies", "each", "all", "both", "versus", "vs", "between", "difference"}


def _search(question: str, sources: list[str] | None, k: int) -> list[str]:
    """Hybrid search inside the given documents (or all). Returns the best k chunk ids."""
    where = None
    if sources and len(sources) == 1:
        where = {"source": sources[0]}
    elif sources:
        where = {"source": {"$in": sources}}

    pool = max(20, 3 * k)
    res = get_collection().query(query_texts=[question], n_results=pool, where=where)
    vector_ids = res["ids"][0]
    keyword_ids = keyword_search(question, sources, pool)

    # Reciprocal rank fusion: chunks ranked high by either method rise to the top.
    fused = Counter()
    # Keyword matches get a little more weight: they're better at exact facts and numbers.
    for ranking, weight in ((vector_ids, 1.0), (keyword_ids, 1.3)):
        for rank, cid in enumerate(ranking):
            fused[cid] += weight / (60 + rank)
    return [cid for cid, _ in fused.most_common(k)]


def retrieve(question: str, k: int = TOP_K):
    """Find the passages most relevant to the question.
    - Names one company  -> search only that document.
    - Names several, or asks to compare -> search each document separately and combine,
      so every company gets its own passages.
    - Otherwise -> search everything."""
    targets = mentioned_sources(question)
    words = set(re.findall(r"[a-z]+", question.lower()))
    q = question.lower()
    asks_which = re.search(r"\bwhich (company|companies|one|firm)\b|\b(most|least|highest|lowest|largest|biggest)\b", q)
    if not targets and (words & COMPARE_WORDS or asks_which):
        targets = list_sources()

    if len(targets) > 1:
        per_doc = max(5, k // len(targets) + 2)
        ids = [cid for src in targets for cid in _search(question, [src], per_doc)]
    else:
        ids = _search(question, targets or None, k)

    c = _load_corpus()
    pos = {cid: i for i, cid in enumerate(c["ids"])}
    return [
        {"text": c["docs"][pos[cid]], "source": c["metas"][pos[cid]]["source"],
         "page": c["metas"][pos[cid]]["page"], "distance": 0.0}
        for cid in ids
    ]


ANSWER_PROMPT = """You are a careful assistant that answers questions using ONLY the context below.

Rules:
- Use only facts found in the context. Do not use outside knowledge.
- After each fact, cite the chunk number in plain square brackets, like [1] or [2][3]. Do not use any other citation style.
- If you calculate something (a percentage, a ratio, a difference), show the numbers you used and the result.
- If only some of the needed facts are in the context, give the ones you found (with citations), say exactly which fact is missing, and don't calculate with a missing number.
- If the question is about several companies, answer for each one the context covers, and say which ones it doesn't cover.
- Different reports use different terms for the same thing (for example "net sales" means revenue).
- Only if the context contains nothing relevant at all, reply exactly: "I couldn't find that in the documents."

Context:
{context}

Question: {question}

Answer:"""


MULTI_QUERY = str(get_setting("MULTI_QUERY", "true")).lower() == "true"

PLAN_PROMPT = """A user asked a question about company annual reports (10-K filings).
Write the separate facts that must be looked up to answer it, as 1 to 6 short search queries, one per line.
Each query must be about ONE company and ONE fact, and must name that company.
Output only the queries, nothing else.

Format: one query per line, written as "<company> <fact>". If the question needs a calculation,
list each input number as its own query. If no company is named, write queries without one.

Question: {question}"""


def plan_queries(question: str) -> list[str]:
    """Split a multi-part question (like revenue per employee) into separate lookups."""
    try:
        lines = generate(PLAN_PROMPT.format(question=question)).splitlines()
    except Exception:
        return []
    queries = [re.sub(r"^[\s\-*\d.)]+", "", l).strip() for l in lines]
    return [q for q in queries if 3 < len(q) < 150][:6]


def gather(question: str) -> tuple[list[dict], list[str]]:
    """Retrieve for the original question, plus each planned sub-query, without duplicates.
    Returns (chunks, sub_queries) so the UI can show what was searched."""
    chunks = retrieve(question)
    subs = []
    if MULTI_QUERY:
        # Sub-queries inherit the company named in the original question.
        targets = mentioned_sources(question)
        for sub in plan_queries(question):
            if targets and not mentioned_sources(sub):
                sub = f"{sub} {os.path.splitext(targets[0])[0]}"
            subs.append(sub)
            chunks += retrieve(sub, k=4)
    seen, unique = set(), []
    for c in chunks:
        key = (c["source"], c["page"], c["text"][:80])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:24], subs


def prepare(question: str) -> dict:
    """Do the retrieval and build the prompt (everything except calling the model)."""
    if get_collection().count() == 0:
        raise RuntimeError("No documents indexed. Put PDFs in the data folder and run: python ingest.py")
    chunks, subs = gather(question)
    context = "\n\n".join(
        f"[{i + 1}] (source: {c['source']}, page {c['page']})\n{c['text']}" for i, c in enumerate(chunks)
    )
    return {"sources": chunks, "context": context, "sub_queries": subs,
            "prompt": ANSWER_PROMPT.format(context=context, question=question)}


def stream_generate(prompt: str):
    """Yield the answer piece by piece as the model writes it. Retries if the server is busy before starting."""
    for attempt in range(4):
        started = False
        try:
            if PROVIDER == "groq":
                stream = get_llm().chat.completions.create(
                    model=CHAT_MODEL, temperature=0, stream=True,
                    messages=[{"role": "user", "content": prompt}],
                )
                for event in stream:
                    piece = event.choices[0].delta.content if event.choices else None
                    if piece:
                        started = True
                        yield piece
            else:
                for event in get_llm().models.generate_content_stream(model=CHAT_MODEL, contents=prompt):
                    if event.text:
                        started = True
                        yield event.text
            return
        except Exception as e:
            if not started and attempt < 3 and ("503" in str(e) or "429" in str(e)):
                time.sleep(5 * (attempt + 1))
            else:
                raise


def answer(question: str) -> dict:
    """Non-streaming version (used by eval.py and the terminal test)."""
    p = prepare(question)
    reply = generate(p["prompt"])
    return {"answer": reply, "sources": p["sources"], "context": p["context"], "sub_queries": p["sub_queries"]}


if __name__ == "__main__":
    # Quick test from the terminal: python rag.py
    q = input("Question: ")
    result = answer(q)
    print("\n" + result["answer"] + "\n")
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] {s['source']} p.{s['page']}")
