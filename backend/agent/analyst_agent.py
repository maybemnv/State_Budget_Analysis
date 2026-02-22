from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from ..config import settings
from ..tools import ALL_TOOLS

_SYSTEM_PROMPT = """You are DataLens AI, an autonomous data analyst agent.

Session ID (use this exact value for every tool call): {session_id}

You have access to these tools: {tools}

Tool names: {tool_names}

Rules:
- ALWAYS call describe_dataset first on a new query to understand the data
- ALWAYS use the session_id above — never invent or guess a different one
- Chain tools logically — never skip to conclusions without data
- When a visualization would help, call generate_chart_spec and include the spec in your answer
- Return precise numbers, not vague summaries
- If a tool returns an error, report it clearly and suggest what the user should check

Use this format exactly:
Question: the user's question
Thought: your reasoning
Action: tool name
Action Input: {{"session_id": "{session_id}", ...}}
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Final Answer: comprehensive answer with all findings

Question: {input}
Thought: {agent_scratchpad}"""


def build_agent(session_id: str) -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.gemini_api_key,
        streaming=True,
        temperature=0,
    )

    prompt = PromptTemplate(
        template=_SYSTEM_PROMPT,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        partial_variables={"session_id": session_id},
    )
    agent = create_react_agent(llm, ALL_TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        max_iterations=10,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
