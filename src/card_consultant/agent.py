from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_card_description
from dotenv import load_dotenv
import os

def main():
    load_dotenv(os.path.join("..", "..", ".env"))

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful credit card consultant. Use tools when needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = create_openai_functions_agent(llm=llm, prompt=prompt, tools=[get_card_description])
    agent_executor = AgentExecutor(agent=agent, tools=[get_card_description], memory=memory, verbose=False)

    print("Ask me anything about credit cards. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = agent_executor.invoke({"input": user_input})
        print(f"\nAssistant: {result['output']}\n")

if __name__ == "__main__":
    main()
