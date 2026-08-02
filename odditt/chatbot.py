"""The RAG chatbot -- works with ANY uploaded PDF(s).

Extracted from the notebook's Section 8 cell. Behavior is unchanged; only the module boundary is
new -- helpers that used to be private functions/classes in neighboring notebook cells
(InMemoryHistory, safe_calculate/CALC_PATTERN, tokenize/cosine_similarity) are now imported from
their own modules instead of living in global notebook scope.
"""
import os
from operator import itemgetter
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path

from .grounding import cosine_similarity, tokenize
from .memory import InMemoryHistory
from .tools import CALC_PATTERN, safe_calculate


class DocChatbot:
    def __init__(self, config: dict, embeddings, llm):
        self.config = config
        self.embeddings = embeddings
        self.llm = llm

        # Chat history store
        self.store = {}

        # Vector database
        self.faiss_index = None

        # Tracks which set of uploaded files the current faiss_index was built from (as a sorted
        # tuple of resolved file paths). Rebuilding is keyed off comparing this to the *current*
        # contents of the uploader on every call -- see process_pdfs / _file_key below. This is
        # what makes the index reflect however many files are currently uploaded, not just
        # whatever was uploaded on the first query of the session.
        self._indexed_file_key = None

        # Cached retrieved documents
        self.last_retrieved_docs = []

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are Odditt, an expert AI document intelligence.\n\n"
                "Your task is to answer questions using ONLY the retrieved document context and the conversation history.\n\n"
                "Instructions:\n"
                "1. Carefully read ALL retrieved document context before answering.\n"
                "2. If the answer is spread across multiple retrieved sections, combine the information into one coherent answer.\n"
                "3. Base every statement on information found in the retrieved document context.\n"
                "4. Do NOT use outside knowledge, prior training, or make up missing facts.\n"
                "5. If only part of the answer is available, answer with the available information and explicitly state which part is not provided in the document.\n"
                "6. If the user asks a follow-up question, use the conversation history to resolve references such as 'it', 'that number', or 'the previous control'.\n"
                "7. If the retrieved context contains evidence that partially answers the question, provide that partial answer instead of replying that you don't know.\n"
                "8. Keep answers concise, factual, and grounded in the retrieved document.\n"
                "9. Each context excerpt below is labeled with its source document filename and page number. "
                "When a question asks you to compare, confirm consistency, or combine facts across more than one "
                "uploaded document, explicitly name which document each fact came from (e.g. 'According to "
                "planning_memo.pdf... while accounting_policies.pdf states...'). Never merge facts from two "
                "different source documents into a single sentence without attributing which came from where.\n"
                "10. If the question requires a calculation — a difference, a percentage, a ratio, an average, "
                "or the gap between two dates — do NOT compute the result yourself. Instead, write it as "
                "CALC[expression], using numbers, the operators + - * / // % ** and parentheses (this covers "
                "virtually any calculation). For dates specifically, a date string like \"2026-01-12\" cannot be "
                "used directly in arithmetic — first convert each date with to_ordinal(\"YYYY-MM-DD\"), which "
                "returns a plain number of days, then do the subtraction/division yourself in the same "
                "expression, e.g. CALC[(to_ordinal(\"2026-02-20\") - to_ordinal(\"2026-01-12\")) / 7] for the "
                "number of weeks between two dates. Example: 'The increase is CALC[4000000 - 3400000] dollars.' "
                "The exact result will be computed and substituted automatically — you do not need to know the "
                "final answer yourself, only the correct expression using values found in the retrieved "
                "context.\n\n"
                "Guardrail vs. unknown — check in this order:\n"
                f"FIRST, if the question is not about the uploaded document(s) at all (general knowledge, personal "
                f"questions about you, chit-chat, or any topic outside the document's subject matter), reply EXACTLY "
                f"with: '{config['guardrail_message']}' — this applies regardless of what was retrieved.\n"
                f"ONLY IF the question IS about the uploaded document(s) but the retrieved context contains no "
                f"information that helps answer it, reply EXACTLY with: '{config['unknown_message']}'"
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Retrieved document context (each excerpt is labeled with its source document and page):\n"
                "{context}\n\n"
                "Question:\n"
                "{query}\n\n"
                "Answer using ONLY the retrieved document context."
            ),
        ])

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryHistory()
        return self.store[session_id]

    def is_no_answer(self, answer: str) -> bool:
        # Detects BOTH guardrail refusals ("I can only answer questions about the uploaded
        # document(s)") and "unknown" replies ("I don't know based on the information...") using
        # key phrases rather than an exact/near-exact string match. Smaller local models
        # frequently paraphrase these instead of reproducing the configured strings verbatim
        # (e.g. "I don't have the information about X in the retrieved document context" instead
        # of the exact configured unknown_message) -- an exact-match check would silently miss
        # those and show a misleading numeric grounding score on what is actually a non-answer.
        a = answer.lower()
        no_answer_phrases = (
            "only answer questions about",
            "don't know based on the information",
            "do not know based on the information",
            "don't have the information",
            "do not have the information",
            "does not contain information",
            "context does not contain",
            "not contain any information",
            "not provided in the given document",
            "not provided in the retrieved document",
            "no information about",
            "no information regarding",
        )
        return any(phrase in a for phrase in no_answer_phrases)

    def _resolve_path(self, pdf_file) -> str:
        # Different Gradio versions hand File/Files components to callbacks differently: older
        # versions pass tempfile-wrapper-like objects with a .name attribute holding the path,
        # current versions (type="filepath", the default) pass plain path strings. Handle both so
        # this doesn't silently break on whichever Gradio version actually gets pip-installed.
        return pdf_file if isinstance(pdf_file, str) else getattr(pdf_file, "name", pdf_file)

    def _file_key(self, pdf_files) -> tuple:
        # A hashable snapshot of exactly which files are currently in the uploader, independent of
        # order. Comparing this on every call is what lets the index track "add a second file",
        # "remove a file", or "swap a file" correctly, instead of only ever reflecting whatever was
        # uploaded on the first query of the session.
        return tuple(sorted(self._resolve_path(f) for f in pdf_files))

    def load_and_chunk(self, pdf_path: str):
        loader = PyMuPDFLoader(pdf_path)
        text_data = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(text_data)

    def ensure_index(self, pdf_files) -> bool:
        # Rebuilds the FAISS index from scratch whenever the *current* set of uploaded files differs
        # from whatever the index currently reflects. This intentionally re-embeds everything (not
        # just newly-added files) on every change — simpler and safer than incremental add, and cheap
        # at the scale of a handful of PDFs per session. Returns False if no readable text could be
        # extracted from any uploaded file (e.g. a scanned/image-only PDF with no text layer), so the
        # caller can show a clear message instead of crashing on a None index.
        current_key = self._file_key(pdf_files)
        if current_key == self._indexed_file_key and self.faiss_index is not None:
            return True

        chunks = []
        for pdf_file in pdf_files:
            chunks.extend(self.load_and_chunk(self._resolve_path(pdf_file)))

        if not chunks:
            self.faiss_index = None
            self._indexed_file_key = None
            return False

        self.faiss_index = FAISS.from_documents(chunks, self.embeddings)
        self._indexed_file_key = current_key
        return True

    def ensemble_retrieve(self, query: str, rrf_k: int = 60):
        # Combine similarity search + MMR search via reciprocal rank fusion.
        # (A small hand-rolled replacement for langchain's EnsembleRetriever, which
        # has moved/been removed across recent LangChain versions.)
        k = self.config["retriever_k"]
        similarity_docs = self.faiss_index.similarity_search(query, k=k)
        mmr_docs = self.faiss_index.max_marginal_relevance_search(query, k=k, fetch_k=40)

        scores = {}
        docs_by_key = {}
        for rank_list in (similarity_docs, mmr_docs):
            for rank, doc in enumerate(rank_list):
                key = (doc.page_content, tuple(sorted(doc.metadata.items())))
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                docs_by_key[key] = doc

        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        top_docs = [docs_by_key[key] for key in ranked_keys[:k]]
        self.last_retrieved_docs = top_docs  # stashed for grounding score + evidence screenshots
        return top_docs

    def format_context(self, docs) -> str:
        # Explicitly labels every retrieved excerpt with its source filename and page, so the model
        # can distinguish facts from different uploaded documents instead of blending them together.
        parts = []
        for d in docs:
            source = d.metadata.get("source") or d.metadata.get("file_path") or "unknown document"
            fname = os.path.basename(source)
            page = d.metadata.get("page", 0)
            parts.append(f"[Source: {fname}, page {page + 1}]\n{d.page_content}")
        return "\n\n".join(parts)

    def compute_grounding_score(self, query: str, answer: str, retrieved_docs) -> int:
        if not retrieved_docs:
            return 0

        # Computed directly via cosine similarity rather than FAISS's built-in relevance_score_fn,
        # which assumes unit-normalized embeddings — sentence-transformers embeddings aren't
        # normalized by default, and using it as-is produces out-of-range, meaningless scores.
        query_vec = self.embeddings.embed_query(query)
        doc_vecs = self.embeddings.embed_documents([d.page_content for d in retrieved_docs])
        relevances = [max(0.0, cosine_similarity(query_vec, v)) for v in doc_vecs]
        avg_relevance = sum(relevances) / len(relevances)

        context_tokens = set()
        for d in retrieved_docs:
            context_tokens |= tokenize(d.page_content)
        answer_tokens = tokenize(answer)
        overlap = (len(answer_tokens & context_tokens) / len(answer_tokens)) if answer_tokens else 0.0

        grounding = 0.5 * avg_relevance + 0.5 * overlap
        return round(grounding * 100)

    def grounding_badge(self, score: int) -> str:
        if score >= 70:
            return f"🟢 **High grounding — {score}%** — answer is well-supported by the retrieved text."
        elif score >= 40:
            return f"🟡 **Medium grounding — {score}%** — spot-check this one against the evidence below."
        else:
            return f"🔴 **Low grounding — {score}%** — verify this answer manually before relying on it."

    def resolve_calculations(self, text: str) -> str:
        def _replace(match):
            raw_expr = match.group(1)
            # The model sometimes copies currency formatting straight out of the source text
            # (e.g. "$4,000,000 - $1,500,000") into the CALC[] expression it writes. "$" and ","
            # aren't valid arithmetic syntax, so ast.parse correctly rejects them -- but nothing
            # was normalizing that cosmetic formatting before handing the expression to the
            # parser. Stripping "$" and "," here doesn't loosen what operations are allowed (the
            # safe-eval grammar is unchanged); it just tolerates the same currency formatting the
            # source PDFs themselves use.
            expr = raw_expr.replace("$", "").replace(",", "").strip()
            try:
                value = safe_calculate(expr)
            except Exception as e:
                return f"[could not compute '{raw_expr}': {e}]"
            if isinstance(value, float):
                return str(int(value)) if value == int(value) else f"{value:.2f}"
            return str(value)
        return CALC_PATTERN.sub(_replace, text)

    def get_source_pages(self, retrieved_docs, max_pages: int = 3):
        seen = set()
        gallery = []
        for d in retrieved_docs:
            source = d.metadata.get("source") or d.metadata.get("file_path")
            page = d.metadata.get("page", 0)
            key = (source, page)
            if not source or key in seen:
                continue
            seen.add(key)
            try:
                images = convert_from_path(source, first_page=page + 1, last_page=page + 1)
            except Exception:
                continue
            if images:
                fname = os.path.basename(source)
                gallery.append((images[0], f"{fname} — page {page + 1}"))
            if len(gallery) >= max_pages:
                break
        return gallery

    def process_pdfs(self, pdf_files, query: str, history: List[dict] = None):
        # Gradio 6 dropped the old [query, answer] "tuples" chatbot format entirely — history is
        # now a flat list of {"role": "user"/"assistant", "content": str} dicts.
        history = history or []
        no_evidence = "_No evidence — no documents uploaded._"
        if not pdf_files:
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "Please upload at least one PDF first."})
            return history, [], no_evidence, ""

        if not self.ensure_index(pdf_files):
            history.append({"role": "user", "content": query})
            history.append({
                "role": "assistant",
                "content": "No readable text could be extracted from the uploaded PDF(s) — they may be "
                           "scanned/image-only pages with no text layer.",
            })
            return history, [], "_No evidence — uploaded PDF(s) contained no extractable text._", ""

        output_parser = StrOutputParser()
        retrieval_chain = (
            {
                "context": itemgetter("query") | RunnableLambda(self.ensemble_retrieve) | RunnableLambda(self.format_context),
                "query": itemgetter("query"),
                "history": itemgetter("history"),
            }
            | RunnableParallel({"output": self.prompt | self.llm | output_parser})
        )

        # Note: you may see a LangChainDeprecationWarning here recommending LangGraph persistence
        # instead. It's safe to ignore -- RunnableWithMessageHistory still works.
        chain_with_history = RunnableWithMessageHistory(
            retrieval_chain,
            self.get_by_session_id,
            input_messages_key="query",
            history_messages_key="history",
        )

        # RunnableWithMessageHistory manages its own internal BaseMessage history (via
        # get_by_session_id) for what the LLM sees — the "history" passed here is unused by it,
        # so we don't need to convert our UI message-dict list back into BaseMessage objects.
        result = chain_with_history.invoke(
            {"query": query, "history": []},
            config={"configurable": {"session_id": "default-session"}},
        )
        answer = self.resolve_calculations(result["output"])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        retrieved_docs = self.last_retrieved_docs or []
        # The evidence gallery now always reflects what was actually retrieved, regardless of
        # whether the answer text happens to match a "no answer" phrase. A no-answer
        # classification only describes what the model *said*, not whether retrieval found
        # anything -- showing the evidence either way lets you tell "the retriever found nothing"
        # apart from "the retriever did its job and the model just didn't use it well".
        gallery = self.get_source_pages(retrieved_docs)

        if self.is_no_answer(answer):
            # Covers both the guardrail case (off-topic question) and the unknown case (on-topic
            # but not found in the documents) -- either way, a numeric grounding score would be
            # misleading here since there's no real "answer" being scored against the retrieved
            # text. The gallery above is still shown, though, so you can check what WAS retrieved.
            grounding_md = "_Grounding score: N/A — no answer given (guardrail or not found in the document(s)). Evidence below shows what was retrieved, if anything._"
        else:
            score = self.compute_grounding_score(query, answer, retrieved_docs)
            grounding_md = self.grounding_badge(score)

        return history, gallery, grounding_md, ""

    def render_file(self, pdf_files):
        if not pdf_files:
            return None
        images = convert_from_path(self._resolve_path(pdf_files[0]), first_page=1, last_page=1)
        return images[0]

    def reset(self):
        self.faiss_index = None
        self._indexed_file_key = None
        self.store = {}
        self.last_retrieved_docs = []
        return [], [], "_No evidence yet — ask a question to see the source page(s) here._", [], None
