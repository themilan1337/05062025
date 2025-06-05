import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from a2a_protocol import A2AMessage, AgentContext, AgentRole, ToolResponse, ConversationTurn

load_dotenv()

# --- LlamaIndex Setup ---
Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

try:
    documents = SimpleDirectoryReader(os.path.join(os.path.dirname(__file__), 'data')).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    print("LlamaIndex Knowledge Agent initialized successfully.")
except Exception as e:
    print(f"Error initializing LlamaIndex: {e}")
    query_engine = None

# --- FastAPI App ---
app = FastAPI()

KNOWLEDGE_AGENT_NAME = "KnowledgeAgentLlama"

@app.post("/a2a_exchange")
async def handle_a2a_message(message: A2AMessage):
    if not query_engine:
        raise HTTPException(status_code=500, detail="Knowledge base not initialized.")

    print(f"\n[{KNOWLEDGE_AGENT_NAME}] Received A2A Message:")
    print(message.model_dump_json(indent=2))

    responses = []

    for turn in message.turns:
        if turn.recipient.agent_name != KNOWLEDGE_AGENT_NAME:
            print(f"Message turn not for me, recipient: {turn.recipient.agent_name}")
            continue

        if turn.tool_calls:
            for tool_call in turn.tool_calls:
                if tool_call.tool_name == "query_knowledge_base":
                    query_text = tool_call.tool_input.get("query")
                    if query_text:
                        print(f"[{KNOWLEDGE_AGENT_NAME}] Processing query: {query_text}")
                        try:
                            response = query_engine.query(query_text)
                            tool_output = {"answer": str(response)}
                            is_error = False
                        except Exception as e:
                            print(f"[{KNOWLEDGE_AGENT_NAME}] Error querying knowledge base: {e}")
                            tool_output = {"error": str(e)}
                            is_error = True

                        tool_response = ToolResponse(
                            tool_name=tool_call.tool_name,
                            tool_call_id=tool_call.tool_call_id,
                            tool_output=tool_output,
                            is_error=is_error
                        )
                        responses.append(tool_response)
                    else:
                        responses.append(ToolResponse(
                            tool_name=tool_call.tool_name,
                            tool_call_id=tool_call.tool_call_id,
                            tool_output={"error": "Missing 'query' in tool_input"},
                            is_error=True
                        ))
                else:
                    responses.append(ToolResponse(
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.tool_call_id,
                        tool_output={"error": f"Unknown tool: {tool_call.tool_name}"},
                        is_error=True
                    ))

    if not responses:
         print(f"[{KNOWLEDGE_AGENT_NAME}] No actionable tool calls found or processed.")

    response_turns = []
    if responses:
        response_turn = ConversationTurn(
            sender=AgentContext(agent_name=KNOWLEDGE_AGENT_NAME, role=AgentRole.TOOL),
            recipient=message.turns[0].sender, # кто прислал первый turn
            tool_responses=responses
        )
        response_turns.append(response_turn)

    if response_turns:
        response_a2a_message = A2AMessage(
            conversation_id=message.conversation_id,
            turns=response_turns
        )
        print(f"[{KNOWLEDGE_AGENT_NAME}] Sending A2A Response:")
        print(response_a2a_message.model_dump_json(indent=2))
        return response_a2a_message
    else:
        return A2AMessage(conversation_id=message.conversation_id, turns=[])


if __name__ == "__main__":
    import uvicorn
    print(f"Starting {KNOWLEDGE_AGENT_NAME} on http://localhost:8001")
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file or environment variables.")
        exit(1)
    if not query_engine:
         print("Knowledge base could not be initialized. Exiting.")
         exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8001)