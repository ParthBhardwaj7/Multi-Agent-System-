"""Database Agent: converts the question into a structured query plan and
executes it through the hardened DatabaseTool.

Security properties:
- The agent's user role is checked against the tool's allowlist BEFORE any
  LLM call (authorization is application code, not the LLM).
- The LLM only proposes a QueryPlan; validation/compilation happens in the tool.
- Any failure yields a structured result — never raw SQL, errors, or secrets."""

from app.agents.base import BaseAgent
from app.core.errors import LLMError
from app.graph.state import GraphState
from app.llm.models import ChatMessage
from app.llm.structured import structured_complete
from app.observability.logging import get_logger
from app.schemas.agents import DatabaseResult, QueryPlan, SpecialistResult
from app.tools.database import TABLE_COLUMNS

logger = get_logger(__name__)


class DatabaseAgent(BaseAgent):
    name = "database"

    def _build_messages(self, state: GraphState) -> list[ChatMessage]:
        from app.core.prompts import load_prompt

        settings_schema = "\n".join(
            f"- {table}({', '.join(sorted(columns))})"
            for table, columns in sorted(TABLE_COLUMNS.items())
        )
        history = "\n".join(
            f"{m.get('role')}: {(m.get('content') or '')[:300]}"
            for m in state.get("messages", [])[-6:]
        )
        context_block = history or "(no prior context)"
        return [
            ChatMessage(role="system", content=load_prompt("database")),
            ChatMessage(
                role="user",
                content=(
                    f"Conversation so far:\n{context_block}\n\n"
                    f"Latest user message:\n{state['current_query']}\n\n"
                    f"Produce a query plan for the latest user message.\n"
                    f"Available tables:\n{settings_schema}"
                ),
            ),
        ]

    async def run(self, state: GraphState) -> GraphState:
        tool = self.context.tools.get("database")
        user_role = state.get("user_role", "user")

        if tool is None:
            result = DatabaseResult(status="unavailable", error_type="tool_missing")
        else:
            # Authorization first — the LLM is never consulted for permission.
            authz = tool.check_authorized(user_role)
            if not authz.allowed:
                logger.info("database_agent_denied", role=user_role)
                result = DatabaseResult(status="denied", reason=authz.reason)
            else:
                try:
                    plan: QueryPlan = await structured_complete(
                        self.provider,
                        QueryPlan,
                        self._build_messages(state),
                        temperature=0.0,
                        max_tokens=512,
                    )
                    result = await tool.execute_plan(plan, user_role=user_role)
                except LLMError as exc:
                    result = DatabaseResult(status="error", error_type=exc.code)

        logger.info("database_agent_completed", status=result.status, rows=result.row_count)
        specialist = SpecialistResult(
            agent=self.name,
            intent="database",
            status=result.status,
            payload=result.model_dump(),
            prompt_version=self.prompt_version(),
        )
        return {"specialist_result": specialist}
