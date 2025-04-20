from langchain_community.tools import tool
from typing import Dict, Any
from db import User, CreditCards, Transaction, Merchant, Owns, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from functools import wraps

db_path = os.path.join("..", "..", "data", "sqlite_database.db")
engine = create_engine(f'sqlite:///{db_path}', echo=False)
Session = sessionmaker(bind=engine)


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