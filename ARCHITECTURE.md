# ClearTrace v0.7.0 — Architecture

## The Idea

One library. Two questions. Every LLM agent you build.

```
  "What did it do?"          "Why did it do that?"
         |                           |
     [ RECORD ]                 [ EXPLAIN ]
         |                           |
         +------ [ IMPROVE ] --------+
```

ClearTrace is the only library that answers both — record the behavior,
explain the reasoning, improve the quality, then regression-test the result.

---

## The DNA

Think of ClearTrace as an organism with three interconnected systems:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   THE NERVOUS SYSTEM          THE BRAIN               THE IMMUNE SYSTEM    ║
║   ──────────────────          ─────────               ─────────────────    ║
║                                                                            ║
║   Senses everything.          Understands why.         Guards against       ║
║   Records every signal        Attributes decisions.    regressions.         ║
║   the agent produces.         Explains behavior.       Catches drift.       ║
║                                                                            ║
║   Recorder                    PromptLIME               Replayer             ║
║   Interceptors                TokenExplainer           TraceDiff            ║
║   Streaming                   CounterfactualGen        Assertions           ║
║   Cassettes                   TrajectoryAttributor     SemanticSimilarity   ║
║   RAG Recorder                ConceptMapper            GapAnalyzer          ║
║   MCP Interceptor             ReasoningEngine          PatternDetector      ║
║                               PromptAdvisor                                ║
║                               ToolAdvisor                                  ║
║                                                                            ║
║   clear_trace.*                 clear_trace.explain.*      clear_trace.*          ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## The Stack

Not a flowchart. A geological cross-section — each layer builds on the one below it.

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │                          SURFACE: WHAT YOU TOUCH                            │
 │                                                                             │
 │  $ cleartrace inspect      $ cleartrace debug       $ cleartrace analyze          │
 │  HTML reports             Time-travel debugger   Word reports                │
 │  Cost dashboards          Matrix heatmaps        Advisor reports             │
 │                                                                             │
 │  cli.py    reporters/html.py    explain/visualization/word_report.py        │
 │                                                                             │
 ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
 │                                                                             │
 │                          INTELLIGENCE LAYER                                 │
 │                                                                             │
 │  ┌─────────────────────┐   ┌──────────────────────┐   ┌────────────────┐   │
 │  │ BEHAVIORAL ANALYSIS │   │  EXPLAIN ENGINE       │   │ QUALITY GATES  │   │
 │  │                     │   │                       │   │                │   │
 │  │ PatternDetector     │   │ PromptLIME            │   │ Budget asserts │   │
 │  │   tool heatmaps     │   │   sentence → impact   │   │ Semantic drift │   │
 │  │   n-gram sequences  │   │ TokenExplainer        │   │ RAG scoring    │   │
 │  │                     │   │   token → impact       │   │ Trace diffing  │   │
 │  │ GapAnalyzer         │   │ Counterfactual        │   │                │   │
 │  │   golden vs actual  │   │   flip → why           │   │ assertions.py  │   │
 │  │   critical gaps     │   │ TrajectoryAttributor  │   │ semantic/      │   │
 │  │                     │   │   step → reason        │   │ rag/scorers    │   │
 │  │ SkillsGenerator     │   │ ConceptMapper         │   │ diff.py        │   │
 │  │   auto AGENTS.md    │   │   behavior → concept   │   │                │   │
 │  │                     │   │ ReasoningEngine       │   │                │   │
 │  │ analysis/           │   │   finding → english    │   │                │   │
 │  └─────────────────────┘   └──────────────────────┘   └────────────────┘   │
 │                                                                             │
 │                            ┌──────────────────────┐                         │
 │                            │  ADVISOR SYSTEM       │                         │
 │                            │                       │                         │
 │                            │  PromptAdvisor        │                         │
 │                            │    5 SE matrices      │                         │
 │                            │    surgical edits     │                         │
 │                            │                       │                         │
 │                            │  ToolAdvisor          │                         │
 │                            │    5 SE matrices      │                         │
 │                            │    schema analysis    │                         │
 │                            │                       │                         │
 │                            │  explain/advisor/     │                         │
 │                            └──────────────────────┘                         │
 │                                                                             │
 ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
 │                                                                             │
 │                          RECORDING LAYER                                    │
 │                                                                             │
 │  ┌───────────────────────────────────────────────────────────────────────┐  │
 │  │                                                                       │  │
 │  │  Recorder ←→ Interceptors ←→ Normalize ←→ Cassette (YAML)           │  │
 │  │      │                                         ↑                      │  │
 │  │      ├── OpenAI    (auto-patched)              │                      │  │
 │  │      ├── Anthropic (auto-patched)        Replayer                     │  │
 │  │      ├── LiteLLM   (auto-patched)              │                      │  │
 │  │      ├── LangChain (callback handler)          │                      │  │
 │  │      ├── LangGraph (Pregel interceptor)   inject recorded             │  │
 │  │      ├── CrewAI    (interceptor)          responses back              │  │
 │  │      ├── RAG       (retrieval events)          │                      │  │
 │  │      └── MCP       (tool call events)    zero API calls               │  │
 │  │                                                                       │  │
 │  │  recorder.py  replayer.py  streaming.py  interceptors/  rag/  mcp/   │  │
 │  └───────────────────────────────────────────────────────────────────────┘  │
 │                                                                             │
 ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
 │                                                                             │
 │                          BEDROCK: TYPES & PRIMITIVES                        │
 │                                                                             │
 │  Trace  TraceEvent  EventType  TraceMetadata       ← _types.py             │
 │  Explanation  TokenImportance  SentenceImportance  ← explain/core/types.py  │
 │  LLMClient  BaseExplainer                          ← explain/core/base.py   │
 │  NormalizedResponse  NormalizedToolCall             ← normalize.py           │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Lifecycle of an Agent Under ClearTrace

A single agent run passes through ClearTrace like light through a prism:

```
                          YOUR AGENT
                              │
                              ▼
                    ┌─────────────────┐
                    │                 │
                    │    RECORD       │   "What happened?"
                    │                 │
                    │  Every LLM call │   Trace with 7 event types:
                    │  Every tool use │   llm_call, tool_call, tool_result,
                    │  Every decision │   agent_decision, retrieval,
                    │  Every chunk    │   mcp_call, error
                    │                 │
                    └────────┬────────┘
                             │
                             ▼
                      cassette.yaml
                    ┌─────────────────┐
                    │ events:         │
                    │   - llm_call    │
                    │     model: gpt4 │
                    │     tokens: 450 │
                    │     cost: $0.01 │
                    │   - tool_call   │
                    │     name: search│
                    │   - llm_call    │
                    │     ...         │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │            │  │            │  │            │
     │   REPLAY   │  │  EXPLAIN   │  │  ANALYZE   │
     │            │  │            │  │            │
     │ Inject     │  │ PromptLIME │  │ Patterns   │
     │ recorded   │  │ Token imp. │  │ Gaps       │
     │ responses  │  │ Counter-   │  │ Skills     │
     │            │  │   factuals │  │            │
     │ Zero cost  │  │ Trajectory │  │ Compared   │
     │ Zero API   │  │ Concepts   │  │ to golden  │
     │ 100ms      │  │ Reasoning  │  │ baselines  │
     │            │  │            │  │            │
     └──────┬─────┘  └─────┬──────┘  └─────┬──────┘
            │              │              │
            ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │            │  │            │  │            │
     │   GUARD    │  │  ADVISE    │  │   STEER    │
     │            │  │            │  │            │
     │ Budget     │  │ Prompt     │  │ AGENTS.md  │
     │ assertions │  │ Advisor    │  │ auto-gen   │
     │ Trace diff │  │ 5 SE       │  │            │
     │ Semantic   │  │ matrices   │  │ Gap        │
     │ drift      │  │            │  │ reports    │
     │ RAG scores │  │ Tool       │  │ for CI     │
     │            │  │ Advisor    │  │            │
     │ pytest     │  │ 5 SE       │  │ exit 1 on  │
     │ native     │  │ matrices   │  │ critical   │
     │            │  │            │  │            │
     └────────────┘  └────────────┘  └────────────┘
```

---

## The Periodic Table of ClearTrace

Every module, classified by function and weight:

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                         PERIODIC TABLE OF CLEARTRACE                              ║
╠═════════════════════════════════════════════════════════════════════════════════  ║
║                                                                                 ║
║  ┌──────┐  RECORD ─────────────────────────────────────────────────────────     ║
║  │ Rc   │  Recorder          The heart. Context manager that monkey-patches     ║
║  │ 39KB │  recorder.py       LLM SDKs and captures every call.                 ║
║  └──────┘                                                                       ║
║  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐        ║
║  │ Rp   │  │ St   │  │ Cs   │  │ Nm   │  │ Rg   │  │ Mc   │  │ Sm   │        ║
║  │ 26KB │  │ 19KB │  │  5KB │  │  4KB │  │ RAG  │  │ MCP  │  │ sem  │        ║
║  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘        ║
║  Replayer  Streaming  Cassette  Normalize  RAG/       MCP       Semantic       ║
║                                            Vector     Tools     Embeddings     ║
║                                                                                 ║
║  ┌──────┐  INTERCEPT ──────────────────────────────────────────────────────     ║
║  │ Oa   │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                   ║
║  │ auto │  │ An   │  │ Lt   │  │ Lc   │  │ Lg   │  │ Cr   │                   ║
║  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘                   ║
║  OpenAI    Anthropic  LiteLLM   LangChain  LangGraph  CrewAI                   ║
║                                                                                 ║
║  ┌──────┐  EXPLAIN ────────────────────────────────────────────────────────     ║
║  │ Lm   │  PromptLIME         Sentence-level perturbation. LIME for LLMs.      ║
║  │ core │  perturbation/                                                        ║
║  └──────┘                                                                       ║
║  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                   ║
║  │ Tk   │  │ Cf   │  │ Ta   │  │ Cx   │  │ Rs   │  │ Cl   │                   ║
║  │ LOO  │  │ flip │  │ traj │  │ map  │  │ why  │  │ base │                   ║
║  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘                   ║
║  Token     Counter-   Trajectory Concept   Reasoning  LLMClient                ║
║  Explainer factual    Attributor Mapper    Engine     BaseExplainer             ║
║                                                                                 ║
║  ┌──────┐  ADVISE ─────────────────────────────────────────────────────────     ║
║  │ Pa   │  PromptAdvisor      5 SE matrices: PCAM, SRM, RGAM, CAR, BAVM       ║
║  │ 5×SE │  advisor/                                                             ║
║  └──────┘                                                                       ║
║  ┌──────┐  ┌──────┐                                                             ║
║  │ Tl   │  │ Mx   │                                                             ║
║  │ 5×SE │  │ heat │                                                             ║
║  └──────┘  └──────┘                                                             ║
║  Tool      Matrix            10 SE quality matrices total                       ║
║  Advisor   Reports           (5 prompt + 5 tool)                                ║
║                                                                                 ║
║  ┌──────┐  ANALYZE ────────────────────────────────────────────────────────     ║
║  │ Pd   │  ┌──────┐  ┌──────┐  ┌──────┐                                        ║
║  │ ptrn │  │ Ga   │  │ Sk   │  │ Gh   │                                        ║
║  └──────┘  └──────┘  └──────┘  └──────┘                                        ║
║  Pattern   Gap       Skills    GitHub                                           ║
║  Detector  Analyzer  Generator PR Fetcher                                       ║
║                                                                                 ║
║  ┌──────┐  GUARD ──────────────────────────────────────────────────────────     ║
║  │ Bg   │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                              ║
║  │ $$$  │  │ Df   │  │ Sd   │  │ Rs   │  │ Pt   │                              ║
║  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘                              ║
║  Budget    Trace     Semantic   RAG       pytest                                ║
║  Asserts   Diff      Drift     Scores    Plugin                                ║
║                                                                                 ║
║  ┌──────┐  SURFACE ────────────────────────────────────────────────────────     ║
║  │ Ht   │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐        ║
║  │ html │  │ Tm   │  │ $d   │  │ Wd   │  │ Hm   │  │ Db   │  │ CLI  │        ║
║  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘        ║
║  HTML      Terminal   Cost     Word       Matrix    Time-      CLI              ║
║  Report    Reporter   Dash     Report     Heatmaps  Travel     (Click)          ║
║                                                     Debugger                    ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════  ╝
```

---

## The File Map

```
src/clear_trace/
│
├── __init__.py                 ← Unified public API (v0.7.0)
├── _types.py                   ← Trace, TraceEvent, EventType, TraceMetadata
├── recorder.py                 ← Recorder context manager (sync + async)
├── replayer.py                 ← Replayer context manager (sync + async)
├── cassette.py                 ← YAML save/load with API key redaction
├── diff.py                     ← Trace diffing engine (DeepDiff)
├── normalize.py                ← Provider-agnostic response normalization
├── assertions.py               ← Budget guards (cost, tokens, loops)
├── streaming.py                ← Stream assembly + realistic chunk replay
├── pytest_plugin.py            ← --record, cassette fixture, @budget marker
├── cli.py                      ← Click CLI: inspect, diff, debug, analyze
│
├── interceptors/               ← Framework monkey-patching
│   ├── langchain.py            ←   LangChain callback handler
│   ├── langgraph.py            ←   LangGraph Pregel interceptor
│   └── crewai.py               ←   CrewAI interceptor
│
├── rag/                        ← RAG recording add-on
│   ├── recorder.py             ←   Retrieval event capture
│   ├── assertions.py           ←   Chunk count, relevance, drift, latency
│   ├── scorers.py              ←   Ragas + DeepEval integrations
│   ├── snapshot.py             ←   RetrieverSnapshot versioning
│   ├── diff.py                 ←   RAG-specific diffing
│   ├── context_analysis.py     ←   Context utilization metrics
│   ├── export.py               ←   RAG dataset export
│   └── interceptors/           ←   ChromaDB, Pinecone, Qdrant, Weaviate, ...
│
├── mcp/                        ← MCP tool call recording
│   ├── interceptor.py          ←   MCP client interceptor
│   ├── events.py               ←   MCP event types
│   └── diff.py                 ←   MCP behavior diffing
│
├── semantic/                   ← Semantic regression detection
│   ├── similarity.py           ←   Embedding-based similarity
│   └── assertions.py           ←   Semantic drift assertions
│
├── analysis/                   ← Behavioral analysis
│   ├── pattern_detector.py     ←   Tool heatmaps, n-gram sequences
│   ├── gap_analyzer.py         ←   Golden baseline comparison
│   └── skills_generator.py     ←   AGENTS.md auto-generation
│
├── github/                     ← GitHub integration
│   └── pr_fetcher.py           ←   PR diff fetching (stdlib only)
│
├── reporters/                  ← Record-side visualization
│   ├── html.py                 ←   Interactive HTML trace reports
│   ├── terminal.py             ←   Rich terminal output
│   └── cost_dashboard.py       ←   Aggregate cost analysis
│
├── export/                     ← Data export
│   └── finetune.py             ←   Fine-tune JSONL (OpenAI/Anthropic)
│
└── explain/                    ← EXPLAINABILITY ENGINE (formerly Prism)
    │
    ├── __init__.py             ← Explain public API
    │
    ├── core/                   ← Foundation
    │   ├── base.py             ←   LLMClient, BaseExplainer
    │   ├── types.py            ←   Explanation, TokenImportance, ...
    │   └── utils.py            ←   Tokenization, similarity, text utils
    │
    ├── perturbation/           ← LIME-style analysis
    │   ├── prompt_lime.py      ←   Sentence-level LIME for prompts
    │   ├── token_importance.py ←   Leave-one-out token analysis
    │   └── sentence_importance.py ← Sentence ablation
    │
    ├── counterfactual/         ← Minimal-edit output flipping
    │   ├── generator.py        ←   Sentence removal, keyword flip, redaction
    │   └── analyzer.py         ←   Sensitivity profiles, robustness scores
    │
    ├── trajectory/             ← Agent decision attribution
    │   ├── attribution.py      ←   TrajectoryAttributor
    │   └── bridge.py  ←   Load cassettes → decisions
    │
    ├── concepts/               ← Concept-based explanations
    │   ├── extractor.py        ←   Marker-based concept detection
    │   └── mapper.py           ←   Ablation-based causal mapping
    │
    ├── reasoning/              ← Natural language explanations
    │   └── engine.py           ←   "Why" explanations for any finding
    │
    ├── advisor/                ← Quality advisory system
    │   ├── advisor.py          ←   PromptAdvisor (analyze, suggest, improve)
    │   ├── suggestions.py      ←   Suggestion data models
    │   ├── matrix_report.py    ←   5 prompt SE matrices (PCAM, SRM, ...)
    │   ├── tool_advisor.py     ←   ToolAdvisor (schema, params, coverage)
    │   ├── tool_types.py       ←   ToolDefinition, ToolReport, ToolIssue
    │   └── tool_matrix_report.py ← 5 tool SE matrices (TSQM, PTAM, ...)
    │
    └── visualization/          ← Explain-side reports
        ├── html_report.py      ←   Interactive HTML explanation reports
        ├── console.py          ←   ConsoleReport, AdvisorReport (Rich)
        ├── matrix_plots.py     ←   Matplotlib matrix heatmaps
        └── word_report.py      ←   Word docs with diff highlighting
```

---

## How They Talk to Each Other

The bridge between Record and Explain is a cassette file:

```
    ┌──────────────┐         cassette.yaml         ┌──────────────────────┐
    │              │  ─────────────────────────────▶│                      │
    │   Recorder   │    events, tools, decisions    │  TrajectoryAttributor│
    │              │                                │  ConceptMapper       │
    │   Records    │                                │  PromptLIME          │
    │   everything │  ◀─────────────────────────────│                      │
    │              │    explain/trajectory/          │  Explains everything │
    └──────────────┘    bridge.py           └──────────────────────┘
                              │
                    load_cleartrace_cassette()
                    cassette_to_decisions()
```

---

## Install What You Need

```
pip install cleartrace                    Core: record, replay, diff, assert
               ╰─[openai]              + OpenAI auto-intercept
               ╰─[anthropic]           + Anthropic auto-intercept
               ╰─[langchain]           + LangChain callback handler
               ╰─[langgraph]           + LangGraph Pregel interceptor
               ╰─[crewai]              + CrewAI interceptor
               ╰─[litellm]             + LiteLLM auto-intercept
               ╰─[rag]                 + RAG event recording
               ╰─[semantic]            + Semantic drift detection
               ╰─[mcp]                 + MCP tool recording
               ╰─[explain]             + PromptLIME, Token, Counterfactual,
               │                         Trajectory, Concepts, Reasoning,
               │                         PromptAdvisor, ToolAdvisor
               ╰─[charts]              + Matplotlib matrix heatmaps
               ╰─[word]                + Word document reports
               ╰─[all]                 Everything above
```

---

## The 10 SE Quality Matrices

The advisor system scores prompts and tools against rigorous
software engineering dimensions:

```
    PROMPT MATRICES                         TOOL MATRICES
    ───────────────                         ─────────────

    ┌──────┐ PCAM                           ┌──────┐ TSQM
    │██████│ Prompt Completeness             │██████│ Tool Schema Quality
    │████░░│ & Adequacy                      │████░░│
    └──────┘                                 └──────┘

    ┌──────┐ SRM                            ┌──────┐ PTAM
    │█████░│ Structural                      │███░░░│ Parameter & Type
    │████░░│ Robustness                      │████░░│ Adequacy
    └──────┘                                 └──────┘

    ┌──────┐ RGAM                           ┌──────┐ TCAM
    │████░░│ Response Guidance               │█████░│ Tool Coverage &
    │███░░░│ & Alignment                     │████░░│ Accessibility
    └──────┘                                 └──────┘

    ┌──────┐ CAR                            ┌──────┐ JCM
    │███░░░│ Compliance &                    │██████│ JSON Compliance
    │██░░░░│ Auditability                    │█████░│
    └──────┘                                 └──────┘

    ┌──────┐ BAVM                           ┌──────┐ CAR
    │████░░│ Behavioral Accuracy             │███░░░│ Compliance &
    │███░░░│ & Validation                    │██░░░░│ Auditability
    └──────┘                                 └──────┘
```

---

## Numbers

```
    81 source files    ·    785+ tests    ·    6 LLM providers
    8 explain modules  ·    10 SE matrices ·   5 report formats
    7 event types      ·    6 vector DBs   ·   1 PyPI package
```
