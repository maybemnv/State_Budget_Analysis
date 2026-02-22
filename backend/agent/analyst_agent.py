from __future__ import annotations

import asyncio
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent import ExceptionTool
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate

from ..config import settings
from ..tools import ALL_TOOLS

_TOOL_GUIDE = """\
Tool selection guide — use the FIRST matching rule:
  Question about schema, columns, data types, or sample rows → describe_dataset
  Means, std, min, max, quartiles for numeric columns → descriptive_stats
  Sums or averages broken down by a category → group_by_stats
  Relationships or correlations between columns → correlation_matrix
  Most common or frequent values in a column → value_counts
  Outliers or anomalies in numeric data → outliers_summary, detect_anomalies
  Reduce dimensions or find structure → run_pca
  Group rows into clusters → run_kmeans
  Predict a numeric value → run_regression
  Predict a category → run_classification
  Trend, seasonality, or stationarity of a time series → check_stationarity, decompose_time_series
  Future values of a time series → run_forecast
  Any chart or visualisation request → generate_chart_spec\
"""

_SYSTEM_PROMPT = """\
You are DataLens AI, an autonomous data analyst agent.

Session ID (pass this exact value as session_id in every tool call): {session_id}

{tool_guide}

Available tools: {tools}
Tool names: {tool_names}

Rules:
- ALWAYS call describe_dataset first on any new query.
- NEVER invent a session_id — use the one above.
- NEVER wrap the Action Input JSON in quotes or markdown code blocks unless the format requires it.
- Chain tools until you have concrete numbers; never skip to conclusions.
- Include a generate_chart_spec call whenever a chart would aid understanding.
- Report tool errors clearly; do not fabricate data.

Use this format exactly:
Question: the input question
Thought: reasoning and which tool to call next
Action: tool_name
Action Input: {{"session_id": "{session_id}", "param": "value"}}
Observation: tool result
...(repeat Thought/Action/Observation)
Thought: I now have enough information to answer.
Final Answer: precise, complete answer with all key numbers

Question: {input}
Thought: {agent_scratchpad}\
"""


def _build_executor(session_id: str) -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.gemini_api_key,
        streaming=True,
        temperature=0,
    )
    prompt = PromptTemplate(
        template=_SYSTEM_PROMPT,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        partial_variables={"session_id": session_id, "tool_guide": _TOOL_GUIDE},
    )
    agent = create_react_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        max_iterations=12,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


@retry(
    retry=retry_if_exception_type(OutputParserException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _invoke(executor: AgentExecutor, payload: dict) -> dict:
    return await executor.ainvoke(payload)


async def run_agent(session_id: str, message: str) -> dict:
    """Build and invoke the agent, returning the raw LangChain result dict.

    Retries up to 3 times on OutputParserException with exponential back-off.
    Returns an error dict on final failure instead of raising.
    """
    executor = _build_executor(session_id)
    try:
        return await _invoke(executor, {"input": message})
    except OutputParserException as e:
        return {"output": f"The agent failed to produce a valid response. Details: {e}", "intermediate_steps": []}
    except Exception as e:
        return {"output": f"An unexpected error occurred: {e}", "intermediate_steps": []}
