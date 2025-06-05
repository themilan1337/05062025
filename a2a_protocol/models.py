from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import uuid
import time

class AgentRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class AgentContext(BaseModel):
    agent_name: str
    role: AgentRole

class ToolCall(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]
    tool_call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ToolResponse(BaseModel):
    tool_name: str
    tool_call_id: str
    tool_output: Any # Может быть строкой, JSON, и т.д.
    is_error: bool = False

class ConversationTurn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    sender: AgentContext
    recipient: AgentContext
    tool_calls: Optional[List[ToolCall]] = None
    tool_responses: Optional[List[ToolResponse]] = None
    text_content: Optional[str] = None
    # Можно добавить другие типы контента: image_content, audio_content и т.д.

class A2AMessage(BaseModel):
    """
    Основное сообщение, которое передается между агентами.
    Может содержать одну или несколько "ходок" диалога.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turns: List[ConversationTurn]

    # Метаданные, если нужны
    metadata: Optional[Dict[str, Any]] = None

# Пример использования (не для файла, просто для понимания)
if __name__ == "__main__":
    # Оркестратор хочет вызвать инструмент у агента знаний
    orchestrator_context = AgentContext(agent_name="OrchestratorAgent", role=AgentRole.ASSISTANT)
    knowledge_agent_context = AgentContext(agent_name="KnowledgeAgent", role=AgentRole.TOOL) # Или ASSISTANT, если он тоже может инициировать

    tool_call = ToolCall(
        tool_name="query_knowledge_base",
        tool_input={"query": "What is the capital of France?"}
    )

    turn_to_knowledge_agent = ConversationTurn(
        sender=orchestrator_context,
        recipient=knowledge_agent_context,
        tool_calls=[tool_call]
    )

    message_to_knowledge_agent = A2AMessage(turns=[turn_to_knowledge_agent])
    print("Message to Knowledge Agent:\n", message_to_knowledge_agent.model_dump_json(indent=2))

    # KnowledgeAgent отвечает
    tool_response = ToolResponse(
        tool_name="query_knowledge_base",
        tool_call_id=tool_call.tool_call_id, # Важно использовать тот же ID
        tool_output={"answer": "Paris"}
    )

    turn_from_knowledge_agent = ConversationTurn(
        sender=knowledge_agent_context,
        recipient=orchestrator_context,
        tool_responses=[tool_response]
    )
    message_from_knowledge_agent = A2AMessage(
        conversation_id=message_to_knowledge_agent.conversation_id, # Та же беседа
        turns=[turn_from_knowledge_agent]
    )
    print("\nMessage from Knowledge Agent:\n", message_from_knowledge_agent.model_dump_json(indent=2))