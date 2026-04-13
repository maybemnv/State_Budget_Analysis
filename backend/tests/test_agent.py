import json
from backend.agent.output_parser import parse_output


class TestOutputParser:
    """Test suite for output parser in agent/output_parser.py"""

    def test_parser_plain_text(self):
        result = parse_output("Revenue peaked at 1.2M in Q3.")
        assert result["answer"] == "Revenue peaked at 1.2M in Q3."
        assert result["chart_spec"] is None
        assert result["has_error"] is False

    def test_parser_extracts_chart_spec(self):
        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "bar",
            "encoding": {"x": {"field": "category"}, "y": {"field": "revenue"}},
        }
        raw = f"Here is the chart:\n```json\n{json.dumps(vega_spec)}\n```\nRevenue is highest in Q3."
        result = parse_output(raw)
        assert result["chart_spec"] == vega_spec
        assert "Revenue is highest in Q3." in result["answer"]
        assert json.dumps(vega_spec) not in result["answer"]

    def test_parser_extracts_chart_spec_without_json_keyword(self):
        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "line",
        }
        raw = f"```\n{json.dumps(vega_spec)}\n```\nSome analysis"
        result = parse_output(raw)
        assert result["chart_spec"] == vega_spec

    def test_parser_flags_error_response_i_encountered(self):
        result = parse_output("I encountered an error: Session not found.")
        assert result["has_error"] is True

    def test_parser_flags_error_response_error(self):
        result = parse_output("Error: Failed to load data")
        assert result["has_error"] is True

    def test_parser_flags_error_response_session(self):
        result = parse_output("Session expired. Please re-upload your data.")
        assert result["has_error"] is True

    def test_parser_no_false_positive_on_normal_text(self):
        result = parse_output("The top category is Electronics with 42% of revenue.")
        assert result["has_error"] is False
        assert result["chart_spec"] is None

    def test_parser_empty_string(self):
        result = parse_output("")
        assert result["answer"] == ""
        assert result["chart_spec"] is None
        assert result["has_error"] is False

    def test_parser_chart_spec_only(self):
        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "scatter",
        }
        raw = f"```json\n{json.dumps(vega_spec)}\n```"
        result = parse_output(raw)
        assert result["chart_spec"] == vega_spec
        assert result["answer"] == ""

    def test_parser_invalid_json_chart_spec(self):
        raw = "```json\n{invalid json}\n```\nSome text"
        result = parse_output(raw)
        assert result["chart_spec"] is None
        assert "Some text" in result["answer"]

    def test_parser_preserves_answer_without_chart(self):
        text = "The average revenue is $1,234.56"
        result = parse_output(text)
        assert result["answer"] == text
        assert result["chart_spec"] is None
        assert result["has_error"] is False
