"""Search Agent: prepares a query via LLM, executes the external search tool,
and returns a structured result. Never generates the final user response."""

from app.agents.base import BaseAgent
from app.core.errors import LLMError, ToolError
from app.graph.state import GraphState
from app.llm.models import ChatMessage
from app.llm.structured import structured_complete
from app.observability.logging import get_logger
from app.schemas.agents import SearchQueryPrep, SearchResult, SpecialistResult
from app.tools.policies import sanitize_for_tool

logger = get_logger(__name__)


class SearchAgent(BaseAgent):
    name = "search"

    def _build_messages(self, state: GraphState) -> list[ChatMessage]:
        from app.core.prompts import load_prompt

        history = "\n".join(
            f"{m.get('role')}: {(m.get('content') or '')[:300]}"
            for m in state.get("messages", [])[-6:]
        )
        context_block = history or "(no prior context)"
        return [
            ChatMessage(role="system", content=load_prompt("search")),
            ChatMessage(
                role="user",
                content=(
                    f"Conversation so far:\n{context_block}\n\n"
                    f"Latest user message:\n{state['current_query']}"
                ),
            ),
        ]

    async def run(self, state: GraphState) -> GraphState:
        tool = self.context.tools.get("search")
        result: SearchResult
        if tool is None:
            result = SearchResult(status="unavailable", error_type="tool_missing")
        else:
            try:
                prepared: SearchQueryPrep = await structured_complete(
                    self.provider,
                    SearchQueryPrep,
                    self._build_messages(state),
                    temperature=0.0,
                    max_tokens=128,
                )
                query = sanitize_for_tool(prepared.query)
                result = await tool.search(query)
            except LLMError as exc:
                result = SearchResult(status="error", error_type=exc.code)
            except ToolError as exc:
                result = SearchResult(status="error", error_type=exc.code)

        logger.info("search_agent_completed", status=result.status, citations=len(result.citations))
        specialist = SpecialistResult(
            agent=self.name,
            intent="search",
            status=result.status,
            payload=result.model_dump(),
            prompt_version=self.prompt_version(),
        )
        return {"specialist_result": specialist}
