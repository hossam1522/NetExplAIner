"""Analyze NetExplAIner results (JSONL or legacy logs).

Same as ``python -m netexplainer analyze``; see ``--help``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netexplainer.evaluation.analysis import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
