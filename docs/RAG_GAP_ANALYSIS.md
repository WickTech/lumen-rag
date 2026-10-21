# Lumen RAG: gap analysis and recruiter-readiness (October 2026)

Method: standard-depth web research (about 14 searches and fetches) plus a read of the repo at `33072ea`.
Primary sources were fetched for Anthropic's Contextual Retrieval post and the Seven Failure Points paper.
Everything else comes from search summaries of secondary sources (vendor blogs, career guides) and is marked
as such. Hiring claims are anecdotal; no source gave data on what actually earns interviews.

## 1. What the research says a RAG pipeline needs

| Stage | Expected capability | Evidence |
|---|---|---|
| Ingestion | Handle new, updated and deleted documents; layout-aware parsing | [ActiveWizards checklist](https://activewizards.com/blog/the-production-ready-rag-pipeline-an-engineering-checklist) (secondary) |
| Chunking | Context-aware chunks | [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (primary) |
| Retrieval | Hybrid dense + BM25, then a reranker | Anthropic: failed top-20 retrievals fell 5.7% to 3.7% with contextual embeddings, 2.9% adding contextual BM25, 1.9% adding reranking (their eval, best embedding config) |
| Query side | Rewriting, decomposition, HyDE, routing | [Practitioner write-ups](https://keywordsai.co/blog/top-8-rag-architectures-to-know-in-2025); no rigorous cross-dataset benchmark found |
| Context assembly | Mind position and size of context | [Lost in the Middle](https://arxiv.org/abs/2307.03172v3): accuracy is worst when relevant text is mid-context; Anthropic found 20 chunks beat 10 or 5 |
| Generation | Grounded, cited, able to say "I don't know" | [Seven Failure Points](https://arxiv.org/html/2401.05856v1) (primary): missing content, missed top-ranked, not in context, not extracted, wrong format, wrong specificity, incomplete |
| Evaluation | Retrieval and answer metrics, regression gates | [Ragas](https://docs.ragas.io/en/v0.1.21/concepts/metrics/): faithfulness, answer relevancy, context precision, context recall |
| Observability | Per-step traces, latency and cost per query, caching | [Jam With AI](https://jamwithai.substack.com/p/production-ready-rag-monitoring-and), [Redis](https://redis.io/blog/rag-at-scale/) (vendor; caching savings figures unverified) |
| Security | Document-level access control, prompt-injection defence | [Wiz OWASP LLM guide](https://www.wiz.io/academy/ai-security/owasp-llm-top-10) (secondary): retrievers have no notion of who may see what; indirect injection via retrieved content |
| Advanced | GraphRAG for corpus-wide or multi-hop questions, behind a router | [Microsoft Research](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/); costly indexing |

## 2. What Lumen already has (genuinely strong)

- Hybrid retrieval (vector, BM25, RRF), optional cross-encoder reranking, Docling layout parsing.
- Offline-deterministic evaluation: recall, MRR, nDCG, per-category metrics, a 176-question v2 set.
- A CI gate and a PR comment that shows metric deltas, claim-level attribution (retrieval vs generation), DeepEval judging.
- pgvector store with namespaces, async API, SSE streaming, OpenTelemetry tracing, rate limiting, a load test, Docker, a hosted HF demo, a changelog and a release workflow.
- Honest reporting: the README publishes a correction where earlier numbers were wrong, and says hybrid underperforms BM25 on the offline embedder. Reviewers read this as maturity.

## 3. Gaps, ranked by recruiter impact

**High (do these first)**
1. **No result on a real embedder.** Every headline number uses the hashing embedder, which the README itself calls weak. Hybrid "losing" to BM25 is an artefact. Re-run the v2 benchmark with a real embedding model (OpenAI, or local bge/e5) and publish both tables. This is the single biggest credibility gap.
2. **No public benchmark.** The eval sets are self-authored (120 short documents). Run on one established set (BEIR subset, HotpotQA, or similar) so the numbers can be compared with others.
3. **No abstention.** There is no relevance threshold: grep finds no `min_score`/refusal logic in the engine. Failure Point 1 (missing content) is the first thing a reviewer will probe. The v2 set already has 23 unanswerable questions, so add a score or LLM-grade gate and report the refusal rate and the false-answer rate on them.
4. **No LLM-generated answer results in the README.** Offline mode is extractive. Show faithfulness, relevancy and context metrics (the DeepEval path exists) on a real model, with cost and latency per query.

**Medium**
5. **No query-side intelligence.** No rewriting, multi-query, decomposition or HyDE (no matches in `lumen_rag/`). Multihop MRR is your weakest category (0.67 vector), so this is the natural thing to improve and measure.
6. **No metadata filtering.** Retrieval cannot filter by source, date or tag, and the API does not expose `namespace`. Namespaces exist in the store only.
7. **No per-user auth or document-level access control.** Only a shared `ADMIN_KEY` on `/reset`. A "multi-tenant" claim needs namespace selection bound to an authenticated identity.
8. **Ingestion is not incremental.** No content hashing, dedup or delete-by-document. Updated files are re-chunked and re-added by hand.
9. **No contextual chunking** (prepend a document-level summary to each chunk) and no parent-document or small-to-big retrieval. Both have the strongest published evidence (Anthropic's numbers above).
10. **No prompt-injection handling** for retrieved text: no instruction hierarchy, no delimiter or quoting policy, no adversarial test cases.
11. **No conversation memory**, so follow-up questions ("and the second one?") fail.

**Lower**
12. No context-budget management (token counting, truncation, ordering for lost-in-the-middle).
13. No semantic cache and no cost or token accounting surfaced in the API.
14. BM25 is rebuilt in-process from stored chunks; Postgres full-text search is documented as future work. Qdrant/Pinecone are not implemented (and honestly stated so).
15. GraphRAG / agentic RAG absent. Skip unless you have a corpus that needs it; it is expensive and easy to over-claim.
16. README inconsistency: the Testing section still says `pytest -q  # 70 tests`, while the status table says 104. Make them agree.

## 4. What recruiters and hiring managers look for

Evidence quality is weak (career blogs, anecdotal), but the sources agree:
([Gauntlet AI](https://gauntletai.com/night-school/what-hiring-managers-actually-look-for-in-ai-engineers),
[MyEngineeringPath](https://myengineeringpath.dev/genai-engineer/portfolio-guide/),
[TechInterview](https://www.techinterview.org/post/3233475399/ai-portfolio-built-with-ai-2026-interviews/),
[Upskillist](https://www.upskillist.com/blog/10-ai-portfolio-examples-impress-recruiters/))

1. **Measured quality**, not a demo: before and after numbers for each change. Lumen's strongest point.
2. **Trade-off reasoning**: why this chunk size, why hybrid, why this reranker, what it cost.
3. **Honest failure analysis**: known limits and what you would do next.
4. **Production signals**: deployed, tested, CI, tracing, rate limits, latency and cost figures.
5. **Scannable in under a minute**: result, diagram and live link at the top. Sources quote scan times from 6 seconds to 8 minutes, so lead with outcomes.
6. **Depth over volume**: one project explained well beats five wrappers; a plain "OpenAI API wrapper" is called too generic.
7. **Judgment about AI itself**: you can say where RAG is the wrong tool.

## 5. Suggested order of work

1. Real-embedder benchmark, both tables, in the README (gap 1).
2. Abstention with refusal and false-answer rates (gap 3).
3. Contextual chunking or query decomposition, each with a before/after table (gaps 9, 5).
4. One public-benchmark run (gap 2).
5. Add a "Limitations and next steps" section and a 60-second outcomes block at the top of the README.
6. Auth-bound namespaces and a prompt-injection test set (gaps 7, 10) to cover the security question.

## 6. Limits of this research

Most hiring and practice claims come from vendor or coaching blogs. Anthropic's percentages are from their own
evaluation and were confirmed on their page, not independently reproduced. The HyDE and Self-RAG original papers were
not opened, so no effect sizes are quoted for them. Repo-gap claims come from keyword searches of `lumen_rag/` and a
read of `llm.py` and `api/app.py`; I did not read every module, so confirm gaps 5, 6, 8 and 12 before acting on them.
