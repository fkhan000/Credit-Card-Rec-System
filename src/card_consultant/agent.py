from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import (get_card_description,
                   #txt_to_sql,
                   compute_savings,
                   get_top_merchants,
                   get_user_profile
                   )
import os
from dotenv import load_dotenv
import streamlit as st



class CreditCardAgent:
    def __init__(self, user_id: int):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        #dotenv_path = os.path.join(base_dir, "..", ".env")
        #dotenv_path = os.path.join("..", ".env")
        #load_dotenv(dotenv_path)
        api_key = st.secrets["OPENAI_API_KEY"]
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        self.user_id = user_id
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        prompt_path = os.path.join(base_dir, "system_prompt.md")

        with open(prompt_path, "r") as f:
            system_prompt = f.read().replace("{user_id}", str(user_id))

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
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
            tools=[get_card_description, compute_savings, get_user_profile, get_top_merchants],
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=[get_card_description, compute_savings, get_user_profile, get_top_merchants],
            memory=self.memory,
            verbose=True,
        )

    def ask(self, user_input: str) -> str:
        result = self.executor.invoke({"input": user_input})
        return result["output"]

    def get_memory(self):
        return self.memory.chat_memory.messages