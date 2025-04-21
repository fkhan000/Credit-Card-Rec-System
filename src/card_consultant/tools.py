from langchain_community.tools import tool
from typing import Dict, Any
from db import User, CreditCards, Transaction, Merchant, Owns, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from functools import wraps
from dotenv import load_dotenv
from sqlalchemy import text
import requests

db_path = os.path.join("..", "..", "data", "sqlite_database.db")
engine = create_engine(f'sqlite:///{db_path}', echo=False)
Session = sessionmaker(bind=engine)

env_path = os.path.join("..", ".env")
#print(os.path.exists(env_path))
#print(os.path.abspath(env_path))
load_dotenv(env_path)
TEXT2SQL_API_KEY = os.getenv("TEXT2SQL_API_KEY")
#print(TEXT2SQL_API_KEY)

@tool
def get_card_description(card_name: str) -> Dict[str, Any]:
    """
    Retrieve the description and benefits of a credit card by name.

    Parameters:
        card_name (str): The name or partial name of the credit card to search for.

    Returns:
    A dictionary with:
        - card_name: The name of the card.
        - description: The card's description.
        - benefits: A list of the card's benefits.

    Example:
        get_card_description({"card_name": "Platinum"})
        {
            "card_name": "Amex Platinum",
            "description": "A premium travel card...",
            "benefits": ["Lounge access", "Concierge service"]
        }
    """
    session = Session()
    try:
        card = session.query(CreditCards).filter(CreditCards.name.ilike(f"%{card_name}%")).first()
        if not card:
            return {"error": f"No credit card found with name containing '{card_name}'."}

        return {
            "card_name": card.name,
            "description": card.description,
            "benefits": card.benefits.split() if isinstance(card.benefits, str) else card.benefits
        }
    finally:
        session.close()

@tool
def txt_to_sql(text_query: str) -> Dict[str, Any]:
    """
    Convert a natural language query into an SQL query using the Text2SQL API,
    execute the generated SQL on the local SQLite database, and return the results.

    Parameters:
        text_query (str): The natural language query to be converted and executed.

    Returns:
    A dictionary with:
        - sql: The generated SQL query.
        - results: The results of the executed SQL query.
        - error (optional): If an error occurs during SQL generation or execution.

    Example:
        txt_to_sql("Find all items with id 1")
        {
            "sql": "SELECT * FROM items WHERE id = 1;",
            "results": [{"id": 1, "name": "Apple"}]
        }
    """
    url = "https://app2.text2sql.ai/api/external/generate-sql"
    headers = {
        "Authorization": f"Bearer {TEXT2SQL_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": text_query,
        "connectionID": "711bb8b8-58d6-4b33-8687-eed63a5e4253"
    }

    try:

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        sql_query = result.get("output")

        if not sql_query:
            return {"error": "No SQL was returned by the API."}

        session = Session()
        try:
            rows = session.execute(text(sql_query)).mappings().all()
            results = [dict(row) for row in rows]
            #columns = rows[0].keys() if rows else []
            #results = [dict(zip(columns, row)) for row in rows]
        finally:
            session.close()

        return {
            "sql": sql_query,
            "results": results
        }

    except requests.RequestException as e:
        return {"error": f"API error: {str(e)}"}
    except Exception as e:
        return {"error": f"SQL execution error: {str(e)}"}