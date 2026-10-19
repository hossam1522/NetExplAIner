"""Registry of the models evaluated in the paper.

Each alias fixes the backend, the backend model name, the context window used
in the experiments and how the trace is rendered (``context``: ``"big"`` keeps
tshark's Info column, ``"small"`` drops it to fit small context windows).

Any name that is not an alias is treated as a raw model name served by the
backend chosen on the command line (``--backend``), with ``context="big"``.
"""

from dataclasses import dataclass, field

from netexplainer.llm.backends import GenerationParams


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    name: str
    backend: str = "ollama"
    context: str = "big"
    params: GenerationParams = GenerationParams()
    extra: dict = field(default_factory=dict)


def _ollama(alias, name, num_ctx=None, context="big", **extra) -> ModelSpec:
    return ModelSpec(
        alias, name, "ollama", context, GenerationParams(num_ctx=num_ctx), extra
    )


def _google(alias, name) -> ModelSpec:
    return ModelSpec(alias, name, "google", "big", GenerationParams(temperature=0))


MODELS: dict[str, ModelSpec] = {
    spec.alias: spec
    for spec in [
        _google("gemini-2.0-flash", "gemini-2.0-flash"),
        _ollama("qwen2.5-32b", "qwen2.5:32b", 32768),
        _google("gemma-3-27b", "gemma-3-27b-it"),
        _ollama("llama2-7b", "llama2", context="small"),
        _ollama("mistral-7b", "mistral", 32768),
        _ollama("llama3.1-8b", "llama3.1", 128000),
        _ollama("gemma-3-12b-ollama", "gemma3:12b", 128000),
        _ollama("granite3.3-8b", "granite3.3:8b", 32000),
        _ollama("phi4", "phi4:14b-fp16", 16000),
        _ollama("gpt-oss-20b", "gpt-oss:20b", 32000),
        _ollama("aya-expanse-8b", "aya-expanse:8b", 8192, context="small"),
        _ollama("deepseek-r1-8b", "deepseek-r1:8b", 32000),
        _ollama("cogito-8b", "cogito:8b", 32000, reasoning=True),
    ]
}


def resolve_model(name: str, default_backend: str = "ollama") -> ModelSpec:
    """Return the registered spec for ``name`` or an ad-hoc spec."""
    if name in MODELS:
        return MODELS[name]
    return ModelSpec(alias=name, name=name, backend=default_backend)
