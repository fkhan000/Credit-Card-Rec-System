from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_card_description
import os
from dotenv import load_dotenv

class CreditCardAgent:
    def __init__(self):
        dotenv_path = os.path.join("..", "..", ".env")
        load_dotenv(dotenv_path)

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful credit card consultant. Use tools when needed."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=[get_card_description],
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=[get_card_description],
            memory=self.memory,
            verbose=True,
        )

    def ask(self, user_input: str) -> str:
        result = self.executor.invoke({"input": user_input})
        return result["output"]

    def get_memory(self):
        return self.memory.chat_memory.messages