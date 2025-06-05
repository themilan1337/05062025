
---

### Как это работает:

1.  **KnowledgeAgent (LlamaIndex + FastAPI):**
 *   Загружает `knowledge.txt` в LlamaIndex VectorStoreIndex.
 *   Запускает FastAPI сервер на `localhost:8001` с эндпоинтом `/a2a_exchange`.
 *   При получении `A2AMessage` на этот эндпоинт:
     *   Проверяет, есть ли `ToolCall` с именем `query_knowledge_base`.
     *   Если да, извлекает `query` из `tool_input`.
     *   Использует LlamaIndex `query_engine` для получения ответа.
     *   Формирует `ToolResponse` и отправляет его обратно в `A2AMessage`.

2.  **OrchestratorAgent (Langchain):**
 *   Инициализируется с `ChatOpenAI` и кастомным инструментом `A2AKnowledgeTool`.
 *   `A2AKnowledgeTool`:
     *   При вызове (_run метод) формирует `A2AMessage` с `ToolCall` для `KnowledgeAgent`.
     *   Отправляет это сообщение POST-запросом на `http://localhost:8001/a2a_exchange`.
     *   Получает ответное `A2AMessage`, парсит `ToolResponse` и возвращает результат (ответ от KnowledgeAgent) как строку.
 *   Когда пользователь вводит запрос (например, "What is the capital of Germany?"):
     *   Langchain агент (благодаря промпту и `create_openai_functions_agent`) решает, что для ответа на этот вопрос нужно использовать `A2AKnowledgeTool`.
     *   Вызывает инструмент, который выполняет A2A коммуникацию.
     *   Получает ответ от инструмента (т.е. от KnowledgeAgent через A2A) и формулирует финальный ответ пользователю.

### Для запуска:

1.  Убедитесь, что `OPENAI_API_KEY` в `.env` корректен.
2.  Откройте два терминала.
3.  В **Терминале 1**:
 ```bash
 # Перейдите в корень проекта, если еще не там
 # cd path/to/multi_agent_a2a_system
 python -m agent_knowledge_llama.main
 ```
 Дождитесь сообщения, что сервер запущен.
4.  В **Терминале 2**:
 ```bash
 # Перейдите в корень проекта, если еще не там
 # cd path/to/multi_agent_a2a_system
 python -m agent_orchestrator_langchain.main
 ```
 Теперь вы можете вводить запросы. Попробуйте:
 *   "What is the capital of France?"
 *   "Tell me about Rome."
 *   "What is the weather in Paris?"
 *   "Who are you?" (должен ответить Orchestrator сам)

Эта реализация демонстрирует основной принцип A2A коммуникации между агентами на разных фреймворках. Ее можно расширять: добавлять больше агентов, усложнять логику, использовать более надежный транспорт сообщений (например, message broker) вместо прямого HTTP для более сложных сценариев.