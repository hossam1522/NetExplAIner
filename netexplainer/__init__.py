"""NetExplAIner: question answering over network traces with LLMs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("netexplainer")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "unknown"
