# NetExplAIner
Using LLMs to analyze network traces by decomposing queries into manageable steps and using mathematical external tools. The idea can be seen more clearly in the following schema:

![image](https://github.com/user-attachments/assets/acbefac2-6951-40e7-b7ad-e127092fa0b7)

The idea consists of investigating the improvement (or not) of the approach of dividing complex questions into manageable steps and providing the LLM model with an external mathematical tool to supply the areas where the LLMs are weaker, like doing math.

## Method

For a question `q` about a trace `T` (rendered as text with `tshark`):

1. **Decomposition**: the LLM generates at most 3 sub-questions `q_1..q_k` (prompt `DECOMPOSITION`).
2. **Answering**: each `q_i` is answered by the LLM looking at the trace: `a_i = LLM(q_i, T)` (prompt `ANSWER`).
3. **Aggregation `g`**: the final answer is synthesized from the question and the pairs only, without the trace: `a = g(q, {(q_i, a_i)})` (prompt `AGGREGATION`, `Pipeline.aggregate` in `netexplainer/llm/pipeline.py`; called `get_final_answer` in earlier versions). The pairs are formatted as `Question i: q_i / Answer i: a_i`.
4. **Judge**: an LLM judge (Gemma 3 27B by default, temperature 0) says whether `a` is almost correct compared with the ground truth computed with scapy (`YES`/`NO`), and scores the similarity (0-100) between the generated and the reference sub-questions of `questions.yaml`.

With **tools**, the calculator (`numexpr`) is bound to every LLM call and the model may call it before answering (at most 10 rounds per call). Tool modes (`tool_mode` in the results), to separate the effect of the tool from the effect of its definition in the prompt:

| Mode | Flag | Behaviour |
|---|---|---|
| `none` | (default) | no tools |
| `callable` | `--tools` | calculator declared and executed |
| `declared` | `--tools-declared-not-callable` | calculator declared (definition bound to the request; described in the ReAct prompt) but never executed: a call gets the constant message `Error: the tool is not available. Answer without using tools.` and is counted in `tool_calls` and `tool_calls_blocked` |

`--tool-ablation` runs `none` and `callable` in the same file; use separate runs (`scripts/run_campaign.sh`) for the three-way ablation. Conditions are labelled `strategy`, `strategy+tools`, `strategy+declared`.

All prompts live in `netexplainer/llm/prompts.py`; a test checks that the ones used in the paper are unchanged.

### Strategies (`--strategy`)

| Strategy | What it does |
|---|---|
| `direct` | answer `q` directly over the trace (no decomposition) |
| `decomp` | full decomposition: steps 1-3 for every question |
| `decomp_tools` | `decomp` with the calculator tool (same as `decomp --tools`) |
| `mixed` | decomposition gated per question: questions in `--direct-questions` (default `1,2,4`: total packets, unique communicators, total bytes) are answered directly, the rest are decomposed. Best configuration of the paper |
| `cot` | zero-shot chain of thought (new, not in the paper): the same question plus "Let's think step by step", ONE call, no expert decomposition (prompt `COT`) |
| `react` | ReAct baseline (new): Thought / Action / Observation loop over the full trace, text-based (`Action: calculator[expr]`), at most `--react-max-steps` calls (default 8; if none gives a `Final Answer` one last call forces it). Every call, token and tool call is recorded; the steps go to `react_steps` |

## Installation

Requirements: Python >= 3.10, [`uv`](https://docs.astral.sh/uv/), `tshark`/`editcap` (Wireshark CLI) and, for local models, [Ollama](https://ollama.com) or an OpenAI-compatible server such as [vLLM](https://docs.vllm.ai).

```bash
make install-uv   # if uv is not installed
make install      # uv sync: exact versions from uv.lock
```

API keys go in a `.env` file (never commit it): `GOOGLE_API_KEY` for Gemini/Gemma (default judge), `OPENAI_API_KEY` for OpenAI-compatible servers that require one (vLLM accepts any value).

Optional plots: `uv sync --extra plots`.

## Data

1. Download the captures from the [Wireshark Wiki](https://wiki.wireshark.org/samplecaptures) into `netexplainer/data/raw/`:
   ```bash
   make download-data
   ```
2. Keep the captures with at most `N` packets in `netexplainer/data/cleaned/`:
   ```bash
   make clean-data N=<N>
   ```
   If you do not want to filter, rename `raw/` to `cleaned/`.

`download-data` fetches exactly the files of `urls.txt` and verifies each one against the SHA-256 in `hashes.txt` (files that do not match are deleted and reported; `--no-verify` skips the check, `--scrape` scrapes the current wiki instead, which is not reproducible). Downloads are retried with backoff (the host answers 429 under load, so expect it to take about an hour) and resumed if interrupted. `.cap` files are converted to `.pcap` after verification.

Questions, reference sub-questions and the default models are in `netexplainer/data/questions.yaml`. A new question also needs its ground truth in `netexplainer/ground_truth.py` (`core_answer` or the `EXTENDED` table).

`make clean-data N=<N>` keeps traces with at most `N` packets in `data/cleaned`; for the scale study use another directory per `N`: `python -m netexplainer clean-data 512 --cleaned-dir netexplainer/data/cleaned512` and then `run --data-dir netexplainer/data/cleaned512 --max-packets 256 --num-ctx 32768` (`--max-packets`/`--min-packets` filter a directory by packet count, e.g. 256 vs 512; the registry `num_ctx` is overridden by `--num-ctx`; each record stores `n_packets`).

### Question sets (`--question-set core|extended|all`)

`core` are the 9 questions of the paper (default). `extended` are Q10-Q15 (`extended_questions` in the YAML), whose ground truth is computed deterministically with scapy in `netexplainer/ground_truth.py`; indexes 10-15 are kept in every set (so `--direct-questions` and routing work unchanged):

| Q | Question | Ground truth |
|---|---|---|
| 10 | TCP/UDP destination port receiving the most packets | port (ties: `p1 or p2`, numeric order) |
| 11 | distinct source IPs sending TCP SYN without ACK | integer |
| 12 | DNS queries (UDP port 53, `qr=0`) | integer; skipped for traces without DNS |
| 13 | IP sending the most bytes (frame length, as tshark shows it) | IP (ties: sorted, ` or `) |
| 14 | host sending to more than 10 distinct TCP/UDP destination ports (scan indicator), multi-hop | sorted IP(s) joined by ` or `, or `none` |
| 15 | TCP packets with RST set | integer; skipped for traces without TCP |

Questions that do not apply to a trace (11, 12, 15 without TCP/DNS) are skipped for it. Extended questions need the Info column (ports, flags), i.e. models with `context="big"` (every alias except `llama2-7b` and `aya-expanse-8b`).

**How numbers and IPs are judged.** The judge prompt of the paper is kept verbatim (`JUDGE_ANSWER`): the judge LLM receives the generated answer and the reference (`str()` of the ground truth) and says whether the answer is "almost correct". There is no numeric tolerance in code: integers (counts, ports) are compared by value, IPs as exact strings, ties as the set of IPs/ports, `none` as the literal absence. Averages and durations (Q5, Q7-Q9) are floats with full precision, so correctness of rounding is left to the judge ("almost correct"); this is a known limitation, quantify it with a second judge (`analyze --second-judge`, per-question agreement and Cohen's kappa) and report it.

## Running

```bash
uv run python -m netexplainer run --help      # or: make run ARGS="..."
```

Examples:

```bash
# Paper setting: full decomposition, phi4 on Ollama, judge Gemma 3 27B (Google API)
uv run python -m netexplainer run --models phi4 --strategy decomp

# Mixed strategy, with and without tools (tool ablation), two models
uv run python -m netexplainer run --models phi4 qwen2.5-32b --strategy mixed --tool-ablation

# Every strategy of the paper at once
uv run python -m netexplainer run --models phi4 --strategy direct decomp mixed --tool-ablation

# Decomposition + tools, custom mixed gating
uv run python -m netexplainer run --models phi4 --strategy decomp_tools
uv run python -m netexplainer run --models phi4 --strategy mixed --direct-questions 1,2,4,5

# vLLM (OpenAI-compatible) model with explicit sampling and a local judge
uv run python -m netexplainer run --backend openai --base-url http://localhost:8000/v1 \
    --models Qwen/Qwen2.5-32B-Instruct --temperature 0 --seed 42 \
    --judge-backend openai --judge-base-url http://localhost:8001/v1 \
    --judge-model google/gemma-3-27b-it

# Generate without judge (e.g. GPUs busy with the model) and judge afterwards
uv run python -m netexplainer run --models phi4 --strategy mixed --judge-backend none --output-dir results/a
uv run python -m netexplainer rejudge results/a/results.jsonl -o results/a/judged.jsonl \
    --judge-backend ollama --judge-model gemma3:27b

# Judge locally with Ollama (gemma3:27b), long CoT/ReAct answers need a larger judge context
uv run python -m netexplainer run --models phi4-q4 --strategy react --tools --question-set all \
    --base-url http://localhost:11437 --judge-backend ollama --judge-model gemma3:27b \
    --judge-base-url http://localhost:11438 --judge-num-ctx 8192

# Tool ablation of the prompt effect (declared in the prompt, never callable)
uv run python -m netexplainer run --models cogito-8b --strategy direct --tools-declared-not-callable

# Quick test on 3 traces, resuming an interrupted run
uv run python -m netexplainer run --models phi4 --max-files 3 --output-dir results/test --resume
```

Models: `--models` accepts the aliases of the paper (`netexplainer/llm/models.py`: `gemini-2.0-flash`, `gemma-3-27b`, `qwen2.5-32b`, `llama2-7b`, `mistral-7b`, `llama3.1-8b`, `gemma-3-12b-ollama`, `granite3.3-8b`, `phi4` (fp16), `phi4-q4` (Q4_K_M, the main runs of the paper), `gpt-oss-20b`, `aya-expanse-8b`, `deepseek-r1-8b`, `cogito-8b`), which fix the backend, model name and context window used in the paper, or any other model name, served by `--backend` (`ollama`, `openai`, `google`) at `--base-url`.

Generation parameters (`--temperature`, `--top-p`, `--top-k`, `--seed`, `--num-ctx`, `--max-tokens`) are only sent when given. By default nothing is sent, which reproduces the paper: Ollama defaults (temperature 0.8, top_p 0.9, top_k 40, unless the model's Modelfile overrides them), temperature 0 for the Google models, and the per-model `num_ctx` of the registry. For vLLM the context length is set when starting the server (`--max-model-len`).

Each question is retried up to `--max-attempts` (default 10) times until the pipeline and the judge calls succeed. Traces are processed in sorted order.

## Results format

`run` appends one JSON object per (model, strategy, tools, trace, question) to `<output-dir>/results.jsonl` (default `results/`). The log goes to `netexplainer/data/evaluation/netexplainer.log` (`--log-file`).

| Field | Description |
|---|---|
| `run_id`, `timestamp`, `netexplainer_version`, `git_commit`, `schema_version` | provenance |
| `model`, `model_name`, `backend` | alias, backend model name and backend |
| `strategy`, `tools`, `condition` | strategy, calculator on/off, and `strategy[+tools]` |
| `file`, `question_index`, `question` | trace and question (1-based index in `questions.yaml`) |
| `decomposed` | whether the question was decomposed |
| `subquestions`, `intermediate_answers` | generated `q_i` and their answers `a_i` |
| `final_answer`, `ground_truth`, `reference_subquestions` | final answer `a`, scapy ground truth, reference sub-questions |
| `judge_model`, `judge_raw`, `verdict`, `correct` | judge output, normalized `YES`/`NO`/`PROBLEM`, `true`/`false`/`null` |
| `subq_similarity_raw`, `subq_similarity` | judge similarity of the sub-questions (0-100, `null` if not decomposed) |
| `attempts`, `errors` | attempts used and error messages of failed attempts |
| `tool_mode`, `n_packets` | `none`/`callable`/`declared` (`tools` is true only for `callable`) and packets of the trace |
| `react_steps` | Thought/Action/Observation steps of `react` (`null` otherwise) |
| `llm_calls`, `tokens_in`, `tokens_out`, `latency_s`, `tool_calls`, `tool_calls_blocked` | usage of the last attempt (tokens as reported by the backend; `tool_calls_blocked` = calls refused in `declared` mode) |
| `seed`, `generation_params` | generation parameters actually sent (`null` = backend default) |

## Analysis

```bash
uv run python -m netexplainer analyze results/results.jsonl --by-question
# or: uv run python scripts/analyze.py results/results.jsonl
```

It prints (markdown, or `--json`):

- accuracy per model and condition (and per question with `--by-question`) with 95% Wilson intervals; `PROBLEM` (judge or pipeline failure) counts as incorrect and is reported in `problems`;
- mean sub-question similarity of decomposed questions;
- exact McNemar tests between every pair of conditions of the same model, paired on (trace, question).

Besides, `analyze` prints (`--latex` for booktabs tabulars, `--json`):

- `accuracy_clustered`: accuracy per model x condition with the Wilson CI and a percentile bootstrap CI that resamples whole traces (`--bootstrap B`, default 1000, `--seed`), since the questions of a trace are not independent;
- `mcnemar_vs_<baseline>_holm`: exact McNemar of every condition against `--baseline` (default `direct`), paired on (trace, question), Holm-corrected over all comparisons;
- `efficiency`: mean LLM calls, tokens in/out, latency and tool calls per condition;
- `judge_agreement` with `--second-judge other.jsonl` (same answers re-judged with `rejudge`): agreement and Cohen's kappa per question.

`--plots DIR` writes the pie, bar and radar charts of the paper (needs the `plots` extra).

### Routing without leakage (`scripts/routing.py`)

Derives the `mixed` policy from `direct` and `decomp` results of the same traces and questions without reusing the evaluation traces (no new LLM calls: mixed answers are taken from the generated ones):

```bash
uv run python scripts/routing.py results/*/results.jsonl --split results/split.json --model phi4-q4
```

1. Deterministic split of the traces into validation 30% / test 70% (`--val-frac`, `--seed`), stratified by number of packets (quartiles); saved in `--split` and reused if it exists.
2. Policy from validation only, per question and per task type (Temporal / Aggregation / Enumeration / Logic&Retrieval, our classification in `TASK_TYPES`, override with `--task-types`): decompose if `ΔAcc = acc(decomp) - acc(direct) > 0` and the lower end of the trace-bootstrap 95% CI of `ΔAcc` is `>= 0` (`--strict`: `> 0`); otherwise `direct`.
3. Test accuracy of `direct`, `decomp`, `mixed_paper` (`--paper-direct`), `mixed_per_question`, `mixed_per_type`, `mixed_loqo` (leave-one-question-out: the policy of the held-out question is the per-type policy derived from the validation items of the other questions of its type; `direct` if none) and `oracle` (best strategy per question on the test set itself, an optimistic bound), with Wilson and trace-bootstrap CIs and paired exact McNemar versus `direct` and `decomp`.

### Campaign

`scripts/run_campaign.sh` runs a models x strategies x tool modes x question sets matrix, one `netexplainer run --resume` per cell in `results/<model>__<strategy>__<tools>__<qset>/results.jsonl` (`.done` marker per finished cell, retries, per-cell log). Ollama for generation on `:11437` and judge on the port you give:

```bash
MODELS="phi4-q4 cogito-8b" STRATEGIES="direct decomp mixed cot react" TOOLS="none callable declared" \
QSETS="core extended" GEN_URL=http://localhost:11437 JUDGE_URL=http://localhost:11438 \
EXTRA="--max-packets 512 --num-ctx 32768" scripts/run_campaign.sh --tmux
```

With `JUDGE_URL=` (empty) the cells are generated without judge; judge them afterwards with `netexplainer rejudge`. After a cut, launch the same command again.

### Re-analyzing the original logs

Runs made before this version only left lines such as `Model: phi4_tools, File: f.pcap, Question: ..., Subquestions Eval: 80, Answer Eval: YES` in `netexplainer.log`. They can be converted to the same format (fields not logged are `null`; the strategy is not in the log, so it must be given):

```bash
uv run python -m netexplainer convert-log old/netexplainer.log -o legacy.jsonl --strategy mixed
uv run python -m netexplainer analyze legacy.jsonl
# or directly: uv run python -m netexplainer analyze old.log --legacy-strategy decomp
```

In the old logs non-decomposed questions were logged with `Subquestions Eval: 100`; they are converted to `subq_similarity: null`.

## Development

```bash
make test     # pytest (LLMs are mocked: no Ollama, GPU or API keys needed)
make lint     # ruff check + format check
make format
```

Code layout:

```
netexplainer/
  cli.py, cli_run.py  command line (run, rejudge, analyze, convert-log, download-data, clean-data)
  dataset.py        pcap -> text with tshark, ground truth with scapy
  scraper.py        Wireshark wiki download and filtering
  ground_truth.py   scapy ground truth of the core and extended questions
  llm/              prompts, backends, model registry, tools, LLM client, strategies
  evaluation/       runner, judge, JSONL records, metrics, stats, routing, analysis, plots
scripts/analyze.py  wrapper of `netexplainer analyze`
scripts/routing.py  leakage-free routing protocol
scripts/run_campaign.sh  experiment matrix with --resume (tmux)
```

## License

[LICENSE](LICENSE)
