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

With **tools**, the calculator (`numexpr`) is bound to every LLM call and the model may call it any number of times before answering.

All prompts live in `netexplainer/llm/prompts.py`; a test checks that the ones used in the paper are unchanged.

### Strategies (`--strategy`)

| Strategy | What it does |
|---|---|
| `direct` | answer `q` directly over the trace (no decomposition) |
| `decomp` | full decomposition: steps 1-3 for every question |
| `decomp_tools` | `decomp` with the calculator tool (same as `decomp --tools`) |
| `mixed` | decomposition gated per question: questions in `--direct-questions` (default `1,2,4`: total packets, unique communicators, total bytes) are answered directly, the rest are decomposed. Best configuration of the paper |
| `cot` | chain-of-thought baseline (new, not in the paper): direct answer asking to think step by step |
| `react` | placeholder, not implemented |

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

Questions, reference sub-questions and the default models are in `netexplainer/data/questions.yaml`. A new question also needs its ground truth in `Dataset.__answer_question` (`netexplainer/dataset.py`).

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
| `llm_calls`, `tokens_in`, `tokens_out`, `latency_s`, `tool_calls` | usage of the last attempt (tokens as reported by the backend) |
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

`--plots DIR` writes the pie, bar and radar charts of the paper (needs the `plots` extra).

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
  cli.py            command line (run, analyze, convert-log, download-data, clean-data)
  dataset.py        pcap -> text with tshark, ground truth with scapy
  scraper.py        Wireshark wiki download and filtering
  llm/              prompts, backends, model registry, tools, LLM client, strategies
  evaluation/       experiment runner, judge, JSONL records, metrics, analysis, plots
scripts/analyze.py  wrapper of `netexplainer analyze`
```

## License

[LICENSE](LICENSE)
