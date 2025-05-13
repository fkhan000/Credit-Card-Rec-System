from langchain_community.tools import tool
from typing import Dict, Any
from db import User, CreditCards, Transaction, Merchant, Owns, Base
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import requests
from datetime import timedelta
# import matplotlib.pyplot as plt
import base64
from io import BytesIO

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

        """
            "grocery_cashback_bonus": card.grocery_cashback_bonus,
            "travel_cashback_bonus": card.travel_cashback_bonus,
            "dining_cashback_bonus": card.dining_cashback_bonus,
            "general_cashback_bonus": card.general_cashback_bonus,
        """
        return {
            "card_name": card.name,
            "description": card.description,
            "benefits": card.benefits.split() if isinstance(card.benefits, str) else card.benefits,
            "grocery_points": int(card.grocery_cashback_bonus / .01),
            "dining_points": int(card.dining_cashback_bonus / .01),
            "travel_points": int(card.travel_cashback_bonus / .01),
            "general_points": int(card.general_cashback_bonus / .01),
            "annual_fee": card.annual_fee            
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
    

@tool
def compute_savings(card_name: str, user_id: int) -> Dict[str, float]:
    """
    Computes the estimated savings for a user over the past 6 months
    based on their transaction history and a specified credit card's rewards program.

    Args:
        card_name (str): The name (or partial name) of the credit card.
        user_id (int): The user id of the user whose transactions will be analyzed.

    Returns:
        Dict[str, float]: A dictionary mapping spending categories ("grocery", "travel", "dining", "general")
                          to the total savings earned in each category based on applicable card bonuses.
    
    Notes:
        - Transactions are filtered to include only those within 6 months of the user's most recent transaction.
        - MCCs (merchant category codes) are used to categorize each transaction.
        - Cashback is calculated using the card's category-specific bonus rates.
    """
    session = Session()

    card = session.query(CreditCards).filter(CreditCards.name.ilike(f"%{card_name}%")).first()

    most_recent_transaction = (
        session.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(Transaction.timestamp.desc())
        .first()
    )

    six_months_ago = most_recent_transaction.timestamp - timedelta(days=6*30)

    recent_transactions = (
        session.query(Transaction)
        .filter(
            Transaction.user_id == user_id,
            Transaction.timestamp >= six_months_ago
        )
        .all()
    )

    savings  = {
        "grocery": 0,
        "travel": 0,
        "dining": 0,
        "general": 0
    }
    for transaction in recent_transactions:
        if transaction.merchant_id in [5411, 5441, 5451, 5462]:
            savings["grocery"] += transaction.amount
        elif transaction.merchant_id in [5812, 5813, 5814]:
            savings["dining"] += transaction.amount
        elif transaction.merchant_id in list(range(3000, 3300)) + [4511]:
            savings["travel"] += transaction.amount
        else:
            savings["general"] += transaction.amount
    
    savings["grocery"] *= card.grocery_cashback_bonus
    savings["general"] *= card.general_cashback_bonus
    savings["travel"] *= card.travel_cashback_bonus
    savings["dining"] *= card.dining_cashback_bonus

    return savings

@tool
def get_user_profile(user_id: int) -> Dict[str, Any]:
    """
    Retrieve the demographic and financial profile of a user by their ID.

    Args:
        user_id (int): The unique identifier of the user in the database.

    Returns:
        Dict[str, Any]: A dictionary containing the user's profile information, including:
            - gender (str): The user's gender (e.g., 'M', 'F').
            - income (float): The user's reported income.
            - date_of_birth (datetime): The user's date of birth.
            - latitude (float): The latitude of the user's home location.
            - longitude (float): The longitude of the user's home location.
            - debt (float): The user's total debt.
            - fico_score (float): The user's FICO credit score.

        If the user is not found, the dictionary will contain an "error" key.

    Example:
        get_user_profile(101) ➝ {
            "gender": "F",
            "income": 85000,
            "date_of_birth": "1991-04-12T00:00:00",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "debt": 12000,
            "fico_score": 720
        }
    """
    session = Session()
    try:
        user = session.query(User).filter(User.user_id == user_id).first()
        if not user:
            return {"error": f"No user found with user id '{user_id}'."}

        return {
            "gender": user.gender,
            "income": user.income,
            "date_of_birth": user.date_of_birth,
            "latitude": user.latitude,
            "longitude": user.longitude,
            "debt": user.debt,
            "fico_score": user.fico_score
        }
    finally:
        session.close()

@tool
def get_top_merchants(user_id: int, n: int = 5) -> Dict[str, Any]:
    """
    Returns the top `n` merchants a user has interacted with based on transaction frequency.

    Args:
        user_id (int): The ID of the user.
        n (int): Number of top merchants to return.

    Returns:
        Dict[str, Any]: A dictionary containing the top merchants and their transaction counts.
        
    Example:
        {
            "top_merchants": [
                {"merchant": "Amazon", "transactions": 12},
                {"merchant": "Starbucks", "transactions": 9},
                {"merchant": "Walmart", "transactions": 7}
            ]
        }
    """
    session = Session()
    try:
        results = (
            session.query(Merchant.description, func.count(Transaction.transaction_id).label("txn_count"))
            .join(Transaction, Transaction.merchant_id == Merchant.merchant_id)
            .filter(Transaction.user_id == user_id)
            .group_by(Merchant.description)
            .order_by(func.count(Transaction.transaction_id).desc())
            .limit(n)
            .all()
        )
        print(results)

        top_merchants = [
            {"merchant": merchant_descriptor, "transactions": txn_count}
            for merchant_descriptor, txn_count in results
        ]

        return {
            "top_merchants": top_merchants
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        session.close()