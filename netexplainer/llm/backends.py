"""Chat model backends: Ollama, OpenAI-compatible servers (vLLM, ...) and Google.

The only interface is :func:`build_chat_model`, which returns a LangChain chat
model. Generation parameters left as ``None`` are NOT sent, so the backend /
model defaults apply. For Ollama this means the defaults used in the paper
(temperature 0.8, top_p 0.9, top_k 40 unless the model's Modelfile overrides
them).
"""

import logging
import os
from dataclasses import asdict, dataclass, fields, replace

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

BACKENDS = ("ollama", "openai", "google")


@dataclass(frozen=True)
class GenerationParams:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    # Context window. Ollama: ``num_ctx``. OpenAI-compatible servers fix it at
    # server start (vLLM ``--max-model-len``), so it is ignored there.
    num_ctx: int | None = None
    max_tokens: int | None = None

    def override(self, other: "GenerationParams") -> "GenerationParams":
        """Return a copy where every non-None field of ``other`` wins."""
        changes = {
            f.name: getattr(other, f.name)
            for f in fields(other)
            if getattr(other, f.name) is not None
        }
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


def _set(kwargs: dict, **values) -> dict:
    kwargs.update({k: v for k, v in values.items() if v is not None})
    return kwargs


def build_chat_model(
    backend: str,
    model: str,
    params: GenerationParams = GenerationParams(),
    base_url: str | None = None,
    api_key: str | None = None,
    extra: dict | None = None,
) -> BaseChatModel:
    """Instantiate the chat model ``model`` served by ``backend``.

    Args:
        backend: one of :data:`BACKENDS`.
        model: model name as known by the backend (e.g. ``qwen2.5:32b``).
        params: generation parameters (``None`` fields are not sent).
        base_url: server URL (Ollama default: ``OLLAMA_HOST`` or localhost;
            OpenAI-compatible: e.g. ``http://localhost:8000/v1``).
        api_key: API key for OpenAI-compatible servers (defaults to
            ``OPENAI_API_KEY`` or ``"EMPTY"``, which vLLM accepts).
        extra: backend specific keyword arguments (e.g. ``reasoning``).
    """
    extra = dict(extra or {})
    p = params
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = _set(
            {"model": model},
            num_ctx=p.num_ctx,
            temperature=p.temperature,
            top_p=p.top_p,
            top_k=p.top_k,
            seed=p.seed,
            num_predict=p.max_tokens,
            base_url=base_url,
        )
        return ChatOllama(**kwargs, **extra)

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        if p.num_ctx is not None:
            logger.warning(
                "num_ctx is ignored by OpenAI-compatible backends; "
                "set the context length when starting the server "
                "(e.g. vLLM --max-model-len)"
            )
        if p.top_k is not None:
            extra.setdefault("extra_body", {})["top_k"] = p.top_k
        kwargs = _set(
            {
                "model": model,
                "api_key": api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            },
            base_url=base_url,
            temperature=p.temperature,
            top_p=p.top_p,
            seed=p.seed,
            max_tokens=p.max_tokens,
        )
        return ChatOpenAI(**kwargs, **extra)

    if backend == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if p.seed is not None or p.num_ctx is not None:
            logger.warning("seed and num_ctx are ignored by the Google backend")
        kwargs = _set(
            {
                "model": model,
                "temperature": p.temperature,
                "max_tokens": p.max_tokens,
                "timeout": None,
            },
            top_p=p.top_p,
            top_k=p.top_k,
        )
        return ChatGoogleGenerativeAI(**kwargs, **extra)

    raise ValueError(f"Unknown backend {backend!r}, expected one of {BACKENDS}")
