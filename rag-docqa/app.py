"""Streamlit chat UI. Run with: streamlit run app.py"""
import glob
import html
import json
import os
import re
import time
from datetime import datetime

import streamlit as st

import ingest
from rag import CHAT_MODEL, get_collection, prepare, stream_generate

st.set_page_config(page_title="DocSage · Document Intelligence", page_icon="📄", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
#MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1050px;}

.hero {padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%); color: white;}
.hero h1 {font-size: 1.7rem; margin: 0; color: white; font-weight: 700;}
.hero p {margin: .35rem 0 0 0; opacity: .88; font-size: .95rem;}

.stat {background: white; border: 1px solid #E2E8F0; border-radius: 10px;
       padding: .7rem .9rem; margin-bottom: .6rem;}
.stat .label {font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: #64748B;}
.stat .value {font-size: 1.25rem; font-weight: 700; color: #0F172A;}

.doc-pill {display: block; padding: .45rem .7rem; margin-bottom: .35rem; border-radius: 8px;
           background: white; border: 1px solid #E2E8F0; font-size: .85rem; color: #334155;
           overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.doc-pill small {color: #94A3B8;}

.cite {display: inline-block; min-width: 1.25rem; padding: 0 .3rem; margin: 0 1px;
       border-radius: 6px; background: #DBEAFE; color: #1D4ED8; font-size: .75rem;
       font-weight: 600; text-align: center; vertical-align: super; line-height: 1.2rem;}

.searched {font-size: .78rem; color: #64748B; margin-bottom: .5rem;}
.chip {display: inline-block; padding: .15rem .55rem; margin: .15rem .25rem .15rem 0; border-radius: 999px;
       background: #F1F5F9; border: 1px solid #E2E8F0; color: #334155; font-size: .76rem;}

.src-card {border: 1px solid #E2E8F0; border-left: 4px solid #2563EB; border-radius: 10px;
           padding: .75rem .9rem; margin-bottom: .6rem; background: #F8FAFC;}
.src-head {display: flex; justify-content: space-between; font-size: .85rem; margin-bottom: .35rem;}
.src-head b {color: #0F172A;}
.src-head span {color: #64748B;}
.src-body {font-size: .82rem; color: #475569; line-height: 1.45;}

.meta {font-size: .78rem; color: #94A3B8; margin-top: .3rem;}
.empty {text-align: center; padding: 2.5rem 1rem 1rem; color: #64748B;}
.empty h3 {color: #0F172A; margin-bottom: .3rem;}

.kpi {background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem 1.1rem;}
.kpi .label {font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: #64748B;}
.kpi .value {font-size: 2rem; font-weight: 700; color: #0F172A; line-height: 1.2;}
.kpi .hint {font-size: .78rem; color: #94A3B8;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Data ----------
@st.cache_resource(show_spinner="First start: indexing the documents (takes a minute or two)...")
def ensure_index() -> bool:
    """On a fresh server (like Streamlit Cloud) there's no database yet, so build it from data/."""
    if get_collection().count() == 0:
        ingest.main()
    return True


ensure_index()
count = get_collection().count()
docs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ingest.DATA_DIR, "*.pdf")))

EXAMPLES = [
    "What is Apple's revenue per employee?",
    "Which company has the highest operating margin?",
    "Which company depends most on a few large customers?",
    "Compare the revenue of the companies.",
]

if "history" not in st.session_state:
    st.session_state.history = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ---------- Helpers ----------
def clean_citations(text: str) -> str:
    """Normalize other citation styles the model sometimes uses: 【2】, 【1†L1-L4】."""
    text = text.replace("​", "")
    return re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", text)


def format_answer(text: str) -> str:
    """Turn [1] citations into small badges."""
    safe = html.escape(clean_citations(text)).replace("\n", "<br>")
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', safe)


def show_searched(subs):
    if subs:
        chips = "".join(f'<span class="chip">🔎 {html.escape(s)}</span>' for s in subs)
        st.markdown(f'<div class="searched">Searched for: {chips}</div>', unsafe_allow_html=True)


def show_sources(sources):
    with st.expander(f"📚 {len(sources)} sources used"):
        for i, s in enumerate(sources, 1):
            snippet = html.escape(s["text"][:420]) + ("…" if len(s["text"]) > 420 else "")
            st.markdown(
                f"""<div class="src-card">
                <div class="src-head"><b>[{i}] {html.escape(s['source'])}</b><span>Page {s['page']}</span></div>
                <div class="src-body">{snippet}</div></div>""",
                unsafe_allow_html=True,
            )


def render_assistant(msg):
    show_searched(msg.get("sub_queries"))
    st.markdown(format_answer(msg["content"]), unsafe_allow_html=True)
    if msg.get("elapsed") is not None:
        st.markdown(f'<div class="meta">Answered in {msg["elapsed"]:.1f}s · {CHAT_MODEL}</div>',
                    unsafe_allow_html=True)
    if msg.get("sources"):
        show_sources(msg["sources"])


def friendly_error(err: str) -> str:
    if "503" in err:
        return "The AI model is busy right now. Please try again in a minute."
    if "429" in err:
        return "The usage limit was reached. Please try again later."
    return f"Something went wrong: {err[:300]}"


# ---------- Export ----------
def chat_markdown(history) -> str:
    lines = [f"# DocQA conversation", f"_Exported {datetime.now():%B %d, %Y %I:%M %p} · model: {CHAT_MODEL}_", ""]
    for m in history:
        if m["role"] == "user":
            lines += [f"## Q: {m['content']}", ""]
        else:
            lines += [clean_citations(m["content"]), ""]
            if m.get("sources"):
                lines.append("**Sources:**")
                lines += [f"- [{i}] {s['source']}, page {s['page']}" for i, s in enumerate(m["sources"], 1)]
                lines.append("")
    return "\n".join(lines)


def _latin(text: str) -> str:
    """The built-in PDF fonts only cover basic characters, so swap the common fancy ones."""
    swaps = {"‑": "-", "–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "≈": "~",
             "×": "x", "÷": "/", "…": "...", "•": "-", "→": "->", "​": "", " ": " ", "\xa0": " "}
    for a, b in swaps.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def chat_pdf(history) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "DocQA conversation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, _latin(f"Exported {datetime.now():%B %d, %Y %I:%M %p} - model: {CHAT_MODEL}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for m in history:
        if m["role"] == "user":
            pdf.set_text_color(30, 58, 138)
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, _latin("Q: " + m["content"]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        else:
            pdf.set_text_color(15, 23, 42)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, _latin(clean_citations(m["content"])), new_x="LMARGIN", new_y="NEXT")
            if m.get("sources"):
                pdf.set_text_color(100, 116, 139)
                pdf.set_font("Helvetica", "", 8)
                refs = "; ".join(f"[{i}] {s['source']} p.{s['page']}" for i, s in enumerate(m["sources"], 1))
                pdf.multi_cell(0, 4, _latin("Sources: " + refs), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
    return bytes(pdf.output())


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📄 DocQA")
    st.caption("Retrieval-augmented Q&A with cited answers")

    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="stat"><div class="label">Documents</div><div class="value">{len(docs)}</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="stat"><div class="label">Chunks</div><div class="value">{count:,}</div></div>',
                unsafe_allow_html=True)

    st.markdown("**Knowledge base**")
    if docs:
        col = get_collection()
        for d in docs:
            n = len(col.get(where={"source": d}, include=[])["ids"])
            note = "⚠️ not indexed" if n == 0 else f"{n:,} chunks"
            st.markdown(f'<span class="doc-pill">📑 {html.escape(d)} <small>· {note}</small></span>',
                        unsafe_allow_html=True)
    else:
        st.caption("No PDFs yet. Upload one below.")

    with st.expander("➕ Add a PDF"):
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key=f"up{st.session_state.uploader_key}",
                                    label_visibility="collapsed")
        if uploaded and st.button("Add to knowledge base", type="primary", use_container_width=True):
            os.makedirs(ingest.DATA_DIR, exist_ok=True)
            path = os.path.join(ingest.DATA_DIR, os.path.basename(uploaded.name))
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner(f"Indexing {uploaded.name}…"):
                pages, n = ingest.index_file(path)
            if n == 0:
                os.remove(path)
                st.error("No text found in that PDF (it may be a scanned image). Try a text-based copy.")
            else:
                st.session_state.flash = f"Added {uploaded.name}: {pages} pages, {n:,} chunks."
                st.session_state.uploader_key += 1
                st.rerun()
        st.caption("Tip: name files by company, like `tesla-10k.pdf`, so questions can target them.")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))

    st.markdown("**Try asking**")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q

    if st.session_state.history:
        st.markdown("**Export conversation**")
        e1, e2 = st.columns(2)
        stamp = f"{datetime.now():%Y%m%d-%H%M}"
        e1.download_button("⬇️ Markdown", chat_markdown(st.session_state.history),
                           file_name=f"docqa-chat-{stamp}.md", mime="text/markdown", use_container_width=True)
        try:
            e2.download_button("⬇️ PDF", chat_pdf(st.session_state.history),
                               file_name=f"docqa-chat-{stamp}.pdf", mime="application/pdf", use_container_width=True)
        except ImportError:
            e2.caption("Run `pip install fpdf2` for PDF export.")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.caption(f"Built with ChromaDB · {CHAT_MODEL} · Streamlit")

# ---------- Main ----------
st.markdown(
    """<div class="hero"><h1>Ask your documents</h1>
    <p>Answers come only from your PDFs, with a citation for every fact. If it's not in the documents, it says so.</p></div>""",
    unsafe_allow_html=True,
)

if count == 0:
    st.warning("No documents indexed yet. Upload a PDF in the sidebar, or add PDFs to the data folder and run: python ingest.py")
    st.stop()

chat_tab, eval_tab = st.tabs(["💬 Ask", "📊 Evaluation"])

# ----- Chat tab -----
with chat_tab:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "📄"):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg)

    typed = st.chat_input("Ask a question about your documents…")
    question = typed or st.session_state.pop("pending", None)

    if not st.session_state.history and not question:
        st.markdown(
            '<div class="empty"><h3>What would you like to know?</h3>'
            "Type a question below, or pick an example from the sidebar.</div>",
            unsafe_allow_html=True,
        )

    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="📄"):
            start = time.time()
            msg = {"role": "assistant", "content": "", "sources": [], "sub_queries": [], "elapsed": None}
            try:
                with st.spinner("Searching your documents…"):
                    prep = prepare(question)
                msg["sources"], msg["sub_queries"] = prep["sources"], prep["sub_queries"]
                show_searched(msg["sub_queries"])

                slot, text = st.empty(), ""
                for piece in stream_generate(prep["prompt"]):
                    text += piece
                    slot.markdown(format_answer(text) + " ▌", unsafe_allow_html=True)
                slot.markdown(format_answer(text), unsafe_allow_html=True)
                msg["content"], msg["elapsed"] = text.strip(), time.time() - start

                st.markdown(f'<div class="meta">Answered in {msg["elapsed"]:.1f}s · {CHAT_MODEL}</div>',
                            unsafe_allow_html=True)
                show_sources(msg["sources"])
            except Exception as e:
                msg["content"] = "⚠️ " + friendly_error(str(e))
                st.markdown(msg["content"])
        st.session_state.history.append(msg)
        st.rerun()  # refresh the sidebar so the export buttons include this answer

# ----- Evaluation tab -----
with eval_tab:
    results_path = "eval_results.json"
    if not os.path.exists(results_path):
        st.info("No evaluation results yet. Write your test questions in `eval_set.json`, then run "
                "`python eval.py` in the terminal and refresh this page.")
    else:
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
        summary, rows = data["summary"], data["results"]
        updated = datetime.fromtimestamp(os.path.getmtime(results_path))
        st.caption(f"{summary['questions']} test questions · last run {updated:%B %d, %Y %I:%M %p}")

        def pct(v):
            return "—" if v is None else f"{v:.0%}"

        kpis = [
            ("Retrieval hit rate", pct(summary.get("retrieval_hit_rate")), "Right document found"),
            ("Answer accuracy", pct(summary.get("answer_accuracy")), "Expected facts in the answer"),
            ("Faithfulness", pct(summary.get("faithfulness")), "No made-up claims"),
            ("Avg latency", "—" if summary.get("avg_latency_s") is None else f"{summary['avg_latency_s']:.1f}s",
             "Time per answer"),
        ]
        for col, (label, value, hint) in zip(st.columns(4), kpis):
            col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                         f'<div class="hint">{hint}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style=\"height:1.2rem\"></div>", unsafe_allow_html=True)
        st.markdown("#### Per-question results")

        def mark(v):
            return "—" if v is None else ("✅ Yes" if v else "❌ No")

        table = [{
            "Question": r["question"],
            "Right doc found": mark(r.get("retrieval_hit")),
            "Answer accuracy": None if r.get("keyword_score") is None else round(r["keyword_score"] * 100),
            "Faithful": mark(r.get("faithful")),
            "Latency (s)": r.get("latency_s"),
        } for r in rows]
        st.dataframe(
            table, use_container_width=True, hide_index=True,
            column_config={
                "Question": st.column_config.TextColumn(width="large"),
                "Answer accuracy": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                "Latency (s)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        misses = [r for r in rows if r.get("faithful") is False or (r.get("keyword_score") or 1) < 1]
        if misses:
            with st.expander(f"🔍 Review {len(misses)} imperfect answers"):
                for r in misses:
                    st.markdown(f"**{r['question']}**")
                    st.markdown(format_answer(r["answer"]), unsafe_allow_html=True)
                    st.caption(f"Judge: {r.get('judge_reason', '')[:300]}")
                    st.divider()
