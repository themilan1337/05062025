import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from .tools import A2AKnowledgeTool, ORCHESTRATOR_AGENT_NAME

load_dotenv()

def run_orchestrator():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file or environment variables.")
        return

    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    tools = [A2AKnowledgeTool()]

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful assistant named {ORCHESTRATOR_AGENT_NAME}. "
                   "You have access to a specialized KnowledgeAgent. "
                   "If a user asks about specific facts like capitals of countries, information about cities, "
                   "or current weather in a specific city that might be in a knowledge base, "
                   "use the 'query_knowledge_agent' tool to get this information. "
                   "Do not make up information if you can retrieve it. "
                   "Always state that you are retrieving information from the KnowledgeAgent when you use the tool. "
                   "If the user asks a general question, or something you can answer directly, do so."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print(f"[{ORCHESTRATOR_AGENT_NAME}] Initialized. Type 'exit' to quit.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        try:
            # response = agent_executor.invoke({"input": user_input})
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            ai_response = response['output']
            print(f"{ORCHESTRATOR_AGENT_NAME}: {ai_response}")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=ai_response))

        except Exception as e:
            print(f"Error during agent execution: {e}")
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=f"Sorry, I encountered an error: {e}"))


if __name__ == "__main__":
    run_orchestrator()