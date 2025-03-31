import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import requests
from load_dotenv import load_dotenv
import requests


def train_test_split():
    transaction_set_filepath = os.path.join("..", "..", "data", "credit_card_transactions-ibm_v2.csv")

    df = pd.read_csv(transaction_set_filepath)

    all_users = np.arange(2000)

    train_users, test_users = train_test_split(all_users, test_size=0.2, random_state=42)

    train_df = df[df['User'].isin(train_users)]
    test_df = df[df['User'].isin(test_users)]

    train_df.to_csv(os.path.join("..", "..", "data", "train_transactions.csv"))
    test_df.to_csv(os.path.join("..", "..", "data", "test_transactions.csv"))

    user_df = pd.read_csv(os.path.join("..", "..", "data", "sd254_users.csv"))

    train_users_df = user_df.iloc[train_users]
    test_users_df = user_df.iloc[test_users]

    train_users_df.to_csv(os.path.join("..", "..", "data", "train_users.csv"))
    test_users_df.to_csv(os.path.join("..", "..", "data", "test_users.csv"))



def get_card_type():
    
    load_dotenv(os.path.join("..", "env"))
    X_API_KEY = os.getenv("X_API_KEY")
    credit_card_filename = os.path.join("..", "..", "data", "sd254_cards.csv")
    credit_card_df = pd.read_csv(credit_card_filename)

    cc_categories = []

    for index, row in credit_card_df.iterrows():
        bin = str(row["Card Number"])[:6]
        bin_lookup_url = 'https://api.api-ninjas.com/v1/bin?bin={}'.format(bin)
        response = requests.get(bin_lookup_url, headers={'X-Api-Key': X_API_KEY})
        if response.status_code == requests.codes.ok:
            try:
                payload = response.json()
                if type(payload) == list:
                    payload = payload[0]
                cc_category = payload.get("category", "UNKNOWN")
            except:
                cc_category = "UNKNOWN"
        else:
            cc_category = "UNKNOWN"
        if index % 20 == 0:
            print(row["Card Brand"], cc_category)
        cc_categories.append(cc_category)

    credit_card_df["Credit Card Category"] = cc_categories
    credit_card_df.to_csv(credit_card_filename)

if __name__ == "__main__":
    #train_test_split()
    get_card_type()