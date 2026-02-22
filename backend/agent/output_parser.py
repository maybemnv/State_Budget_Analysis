import json
import re
from typing import TypedDict


class AnalysisResult(TypedDict):
    answer: str
    chart_spec: dict | None
    has_error: bool


_CHART_BLOCK = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\"\\$schema\"[\s\S]*?\})\s*```", re.IGNORECASE)
_ERROR_PREFIXES = ("i encountered an error", "error:", "session")


def parse_output(raw: str) -> AnalysisResult:
    chart_spec: dict | None = None
    answer = raw.strip()

    match = _CHART_BLOCK.search(answer)
    if match:
        try:
            chart_spec = json.loads(match.group(1))
            answer = (answer[: match.start()] + answer[match.end() :]).strip()
        except json.JSONDecodeError:
            pass

    has_error = any(answer.lower().startswith(p) for p in _ERROR_PREFIXES)

    return AnalysisResult(answer=answer, chart_spec=chart_spec, has_error=has_error)
