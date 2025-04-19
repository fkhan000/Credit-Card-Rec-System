import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import requests
from load_dotenv import load_dotenv
import requests
import replicate
import json
import numpy as np
from sklearn.cluster import KMeans
import pickle
from tqdm import tqdm

def get_card_type():
    
    load_dotenv(os.path.join("..", "..", "env"))
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
    credit_card_df.to_csv(os.path.join("..", "..", "data", "cc_dataset_with_labels.csv"))

def get_merchant_embeddings():
    load_dotenv(os.path.join("..", "..", "env"))
    api = replicate.Client(api_token=os.getenv("REPLICATE_API_KEY"))

    with open(os.path.join("..", "..", "data", "mcc_description.json")) as f:
        merchants = json.load(f)
    for index in tqdm(range(len(merchants))):
        embedding = api.run(
                    "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
                    input={"text_input": merchants[index]["description"], "modality": "text"},
                )
        merchants[index]["embedding"] = list(embedding / np.sqrt(np.dot(embedding, embedding)))
    
    with open(os.path.join("..", "..", "data", "merchant_data.json"), "w") as json_file:
        json.dump(merchants, json_file)

def generate_clustering_model(n_clusters):
    merchant_data_file = os.path.join("..", "..", "data", "merchant_data.json")
    if not os.path.exists(merchant_data_file):
        get_merchant_embeddings()

    with open(merchant_data_file, "r") as f:
        merchants = json.load(f)
    X = np.array([merchant["embedding"] for merchant in merchants])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    with open(os.path.join("..", "..", "models", "merchant_clustering_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    labels = kmeans.predict(X)
    print(kmeans.score(X))
    for index in range(len(merchants)):
        merchants[index]["category"] = int(labels[index])
    print(merchants[0])
    with open(merchant_data_file, "w") as f:
        json.dump(merchants, f)

def create_merchant_categories(n_clusters, generate=True):
    if generate:
        generate_clustering_model(n_clusters=n_clusters)
    with open(os.path.join("..", "..", "data", "merchant_data.json"), "r") as f:
        merchants = json.load(f)
    
    mcc_to_category = {merchant["mcc"]: merchant["category"] for merchant in merchants}
    return mcc_to_category
    

def get_ratings():

    transaction_set_filepath = os.path.join("..", "..", "data", "credit_card_transactions-ibm_v2.csv")
    cc_set_filepath = os.path.join("..", "..", "data", "cc_dataset_with_labels.csv")

    transaction_df = pd.read_csv(transaction_set_filepath, usecols=["User", "Card"])
    cc_df = pd.read_csv(cc_set_filepath)

    transaction_df.rename(columns={"Card": "CARD INDEX"}, inplace=True)

    card_map = cc_df.set_index("CARD INDEX")["Card Category"].to_dict()

    transaction_df["Card Category"] = transaction_df["CARD INDEX"].map(card_map)

    transaction_df.dropna(subset=["Card Category"], inplace=True)

    grouped = transaction_df.groupby(['User', 'Card Category']).size().reset_index(name='TxCount')
    total = transaction_df.groupby('User').size().reset_index(name='TotalTx')

    ratings = pd.merge(grouped, total, on='User')
    ratings['Rating'] = ratings['TxCount'] / ratings['TotalTx']

    cc_df = cc_df.merge(ratings, on=['User', 'Card Category'], how='left')
    cc_df['Rating'] = cc_df['Rating'].fillna(0.0)

    cc_df.to_csv(cc_set_filepath, index=False)



def prepare_transaction_data(n_clusters):
    transaction_set_filepath = os.path.join("..", "..", "data", "credit_card_transactions-ibm_v2.csv")

    df = pd.read_csv(transaction_set_filepath)
    mcc_to_category = create_merchant_categories(n_clusters, generate=True)
    df["Merchant Category"] = df["MCC"].apply(lambda x: mcc_to_category.get(x, -1))

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

if __name__ == "__main__":
    #create_merchant_categories(10)
    #get_card_type()
    prepare_transaction_data(10, generate=False)
    #get_ratings()