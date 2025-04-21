from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    TIMESTAMP
)
from sqlalchemy.orm import declarative_base, sessionmaker
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json

Base = declarative_base()

class User(Base):
    """
    Represents a user in the system.

    Attributes:
        id (int): The primary key for the user.
        gender (str): The gender of the user (e.g., 'M', 'F', etc.).
        income (int): The income of the user.
        date_of_birth (timestamp): The date the user was born.
        latitude (float): The latitude coordinate of the user's home.
        longitude (float): The longitude coordinate of the user's home.
        debt (float): How much debt the user has.
        fico_score (float): A user's fico score
    """
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True)
    gender = Column(String, nullable=False)
    income = Column(Float, nullable=False)
    date_of_birth = Column(TIMESTAMP, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    debt = Column(Float, nullable=False)
    fico_score = Column(Float, nullable=False)

class Transaction(Base):
    """
    Represents a financial transaction made by a user.

    Attributes:
        transaction_id (int): The primary key for the transaction.
        user_id (int): Foreign key referencing the user who made the transaction.
        cc_id (int): Foreign key referencing the credit card used for the transaction.
        merchant_id (int): Foreign key referencing the merchant where the transaction occurred.
        amount (float): The transaction amount.
        zipcode (int): The ZIP code where the transaction occurred.
        timestamp (datetime): The timestamp of the transaction.
    """
    __tablename__ = 'transactions'

    transaction_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    cc_id = Column(Integer, ForeignKey('credit_cards.cc_id'))
    merchant_id = Column(Integer, ForeignKey('merchants.merchant_id'))
    amount = Column(Float, nullable=False)
    zipcode = Column(Integer, nullable=True)
    timestamp = Column(TIMESTAMP, nullable=False)


class CreditCards(Base):
    """
    Represents a credit card.

    Attributes:
        cc_id (int): The primary key for the credit card.
        name (str): The name of the card.
        description (str): A description of the credit card (e.g., 'Visa', 'MasterCard').
        benefits (str): A description of the benefits that the card provides.
    """
    __tablename__ = 'credit_cards'

    cc_id = Column(Integer, primary_key=True)
    name=Column(String, nullable=False)
    grocery_cashback_bonus = Column(Float, nullable=False)
    travel_cashback_bonus = Column(Float, nullable=False)
    dining_cashback_bonus = Column(Float, nullable=False)
    general_cashback_bonus = Column(Float, nullable=False)
    annual_fee = Column(Float, nullable=False)
    description = Column(String, nullable=False)
    benefits = Column(String, nullable=False)


class Owns(Base):
    """
    Represents the ownership relationship between a user and a credit card.

    Attributes:
        user_id (int): The primary key referencing a user.
        cc_id (int): The primary key referencing a credit card.
    """
    __tablename__ = 'owns'

    user_id = Column(Integer, ForeignKey('users.user_id'), primary_key=True)
    cc_id = Column(Integer, ForeignKey('credit_cards.cc_id'), primary_key=True)


class Merchant(Base):
    """
    Represents a merchant.

    Attributes:
        merchant_id (int): The primary key for the merchant.
        description (str): A description of the merchant (e.g., 'Amazon', 'Walmart').
    """
    __tablename__ = 'merchants'

    merchant_id = Column(Integer, primary_key=True)
    description = Column(String, nullable=False)


def main():
    db_path = os.path.join("..", "..", "data", "sqlite_database.db")
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    #engine = create_engine('sqlite:///:db::', echo=False)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    print("Loading in users ...")
    users_df = pd.read_csv(os.path.join("..", "..", "data", "sd254_users.csv"))
    for _, row in tqdm(users_df.iterrows(), total=len(users_df)):
        new_user = User(
            gender=row["Gender"],
            income=float(row["Yearly Income - Person"][1:]),
            date_of_birth=datetime(row["Birth Year"], row["Birth Month"], 1),
            latitude=row["Latitude"],
            longitude=row["Longitude"],
            debt=float(row["Total Debt"][1:]),
            fico_score=row["FICO Score"]
        )
        session.add(new_user)
    
    print("Loading in credit cards...")
    #TODO: Make the credit_cards.json file
    with open(os.path.join("..", "..", "data", "credit_cards.json")) as f:
        cc_dict = json.load(f)["credit_cards"]
    
    for card in tqdm(cc_dict):
        new_cc = CreditCards(
            name=card["name"],
            description=card["description"],
            grocery_cashback_bonus = card["grocery_cashback_bonus"],
            dining_cashback_bonus = card["dining_cashback_bonus"],
            travel_cashback_bonus = card["travel_cashback_bonus"],
            general_cashback_bonus = card["general_cashback_bonus"],
            annual_fee = card["annual_fee"],
            benefits=" ".join(card["benefits"])
        )
        session.add(new_cc)
    
    print("\nLoading in Merchants...")
    with open(os.path.join("..", "..", "data", "mcc_description.json")) as f:
        merchant_dict = json.load(f)
    
    for merchant in tqdm(merchant_dict):
        new_merchant = Merchant(
            merchant_id=merchant["mcc"],
            description=merchant["description"]
        )
        session.add(new_merchant)

    print("\nLoading in Transactions...")
    
    cc_df = pd.read_csv(os.path.join("..", "..", "data", "cc_dataset_with_labels.csv"))
    cc_to_category = dict(zip(cc_df["CARD INDEX"],
                              cc_df["Card Category"]))
    category_to_chase = {
        "Visa Classic": "Chase Freedom Flex",
        "Visa Gold": "Chase Slate Edge",
        "Visa Platinum": "Chase Sapphire Preferred",
        "Mastercard Standard": "Chase Freedom Unlimited",
        "Mastercard Gold": "Chase Sapphire Preferred",
        "Mastercard Platinum": "Chase Sapphire Preferred",
        "Amex Blue": "Chase Freedom",
        "Amex Gold": "Chase Sapphire Preferred",
        "Amex Platinum": "Chase Saphire Reserve"
    }

    categories = ["Chase Freedom Flex", "Chase Slate Edge", "Chase Sapphire Preferred",
                  "Chase Freedom Unlimited", "Chase Freedom", "Chase Sapphire Reserve"]

    category_to_id = (dict(zip
                           (
                               categories,
                               list(range(1, len(categories) + 1)),                          
                      )))
    
    transactions_df = pd.read_csv(
        os.path.join("..", "..", "data", "credit_card_transactions-ibm_v2.csv"),
        usecols=["User", "Card", "MCC", "Year", "Month", "Day", "Zip", "Amount"]
        )
    for index, row in tqdm(transactions_df.iterrows(), total=len(transactions_df)):
        if index > 1e6:
            break

        new_transaction = Transaction(
            user_id = row["User"],
            cc_id = category_to_id[category_to_chase[cc_to_category[row["Card"]]]],
            merchant_id = row["MCC"],
            amount = float(row["Amount"][1:]),
            zipcode = row["Zip"],
            timestamp = datetime(row["Year"], row["Month"], row["Day"])
        )
        session.add(new_transaction)
    session.commit()




if __name__ == '__main__':
    main()

