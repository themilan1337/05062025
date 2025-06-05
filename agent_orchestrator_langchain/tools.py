import requests
import json
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any

from a2a_protocol import A2AMessage, AgentContext, AgentRole, ToolCall, ConversationTurn

# Имена агентов должны быть консистентны
ORCHESTRATOR_AGENT_NAME = "OrchestratorAgentLangchain"
KNOWLEDGE_AGENT_NAME = "KnowledgeAgentLlama" # Должно совпадать с тем, что в KnowledgeAgent
KNOWLEDGE_AGENT_URL = "http://localhost:8001/a2a_exchange" # URL нашего KnowledgeAgent

class KnowledgeBaseQueryInput(BaseModel):
    query: str = Field(description="The query to send to the knowledge base agent.")

class A2AKnowledgeTool(BaseTool):
    name: str = "query_knowledge_agent"
    description: str = (
        "Use this tool to query the KnowledgeAgent for specific information. "
        "Input should be a natural language query."
    )
    args_schema: Type[BaseModel] = KnowledgeBaseQueryInput

    def _parse_a2a_response(self, response_data: Dict[str, Any]) -> str:
        """Парсит ответ A2A и извлекает полезную информацию."""
        try:
            a2a_response = A2AMessage.model_validate(response_data)
            results = []
            for turn in a2a_response.turns:
                if turn.tool_responses:
                    for tool_resp in turn.tool_responses:
                        if tool_resp.tool_name == "query_knowledge_base": # Ожидаемое имя от KnowledgeAgent
                            if not tool_resp.is_error:
                                # Предполагаем, что ответ в tool_output.answer
                                answer = tool_resp.tool_output.get("answer", "No answer found.")
                                results.append(str(answer)) # Убедимся, что это строка
                            else:
                                error_msg = tool_resp.tool_output.get("error", "Unknown error from tool.")
                                results.append(f"Error from KnowledgeAgent: {error_msg}")
            return "\n".join(results) if results else "KnowledgeAgent provided no usable response."
        except Exception as e:
            print(f"Error parsing A2A response: {e}")
            return f"Error parsing response from KnowledgeAgent: {str(e)}"

    def _run(self, query: str) -> str:
        """Используется Langchain для выполнения инструмента."""
        print(f"\n[{ORCHESTRATOR_AGENT_NAME}] Using A2AKnowledgeTool with query: {query}")

        # 1. Создаем A2A сообщение
        orchestrator_context = AgentContext(agent_name=ORCHESTRATOR_AGENT_NAME, role=AgentRole.ASSISTANT)
        knowledge_agent_context = AgentContext(agent_name=KNOWLEDGE_AGENT_NAME, role=AgentRole.TOOL) # или ASSISTANT

        tool_call = ToolCall(
            tool_name="query_knowledge_base", # Это имя инструмента, которое ожидает KnowledgeAgent
            tool_input={"query": query}
        )

        turn_to_knowledge_agent = ConversationTurn(
            sender=orchestrator_context,
            recipient=knowledge_agent_context,
            tool_calls=[tool_call]
        )

        message_to_knowledge_agent = A2AMessage(turns=[turn_to_knowledge_agent])

        # 2. Отправляем HTTP POST запрос
        try:
            print(f"[{ORCHESTRATOR_AGENT_NAME}] Sending A2A message to {KNOWLEDGE_AGENT_URL}:")
            print(message_to_knowledge_agent.model_dump_json(indent=2))

            response = requests.post(
                KNOWLEDGE_AGENT_URL,
                json=message_to_knowledge_agent.model_dump() # Pydantic .model_dump() для сериализации
            )
            response.raise_for_status() # Вызовет исключение для HTTP ошибок (4xx, 5xx)
            
            response_data = response.json()
            print(f"[{ORCHESTRATOR_AGENT_NAME}] Received A2A response from KnowledgeAgent:")
            print(json.dumps(response_data, indent=2))

            # 3. Парсим ответ
            parsed_output = self._parse_a2a_response(response_data)
            return parsed_output

        except requests.exceptions.RequestException as e:
            print(f"[{ORCHESTRATOR_AGENT_NAME}] HTTP Request failed: {e}")
            return f"Error communicating with KnowledgeAgent: {e}"
        except Exception as e:
            print(f"[{ORCHESTRATOR_AGENT_NAME}] Error processing A2A exchange: {e}")
            return f"Unexpected error during A2A exchange: {e}"

    async def _arun(self, query: str) -> str:
        # Для асинхронного выполнения, если потребуется. Пока не реализуем.
        # Можно использовать aiohttp для асинхронных запросов.
        raise NotImplementedError("A2AKnowledgeTool does not support async operation yet.")