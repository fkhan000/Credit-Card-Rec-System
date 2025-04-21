from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_card_description, txt_to_sql
import os
from dotenv import load_dotenv

class CreditCardAgent:
    def __init__(self):
        dotenv_path = os.path.join("..", ".env")
        load_dotenv(dotenv_path)

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        schema_description = """
You are a helpful credit card consultant. You have access to tools to answer questions.
You can also use a text-to-SQL tool to query a database.

Here is the database schema you can use when appropriate:
User(id: int, name: text, date_of_birth: text, gender: text, income: float, latitude: float, longitude: float, fico_score: float, debt: float)
CreditCards(id: int, name: text, description: text, benefits: text)
Transaction(id: int, user_id: int, merchant_id: int, card_id: int, amount: float, date: text)
Merchant(id: int, name: text, category: text)
Owns(user_id: int, card_id: int)

Use the `txt_to_sql` tool if the user asks about anything that involves searching or analyzing user, card, transaction, or merchant information.
        """.strip()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", schema_description),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=[get_card_description, txt_to_sql],
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=[get_card_description, txt_to_sql],
            memory=self.memory,
            verbose=True,
        )

    def ask(self, user_input: str) -> str:
        result = self.executor.invoke({"input": user_input})
        return result["output"]

    def get_memory(self):
        return self.memory.chat_memory.messages