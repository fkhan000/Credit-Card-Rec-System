import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd


transaction_set_filepath = os.path.join("..", "data", "credit_card_transactions-ibm_v2.csv")

df = pd.read_csv(transaction_set_filepath)

all_users = np.arange(2000)

train_users, test_users = train_test_split(all_users, test_size=0.2, random_state=42)

train_df = df[df['User'].isin(train_users)]
test_df = df[df['User'].isin(test_users)]

train_df.to_csv(os.path.join("..", "data", "train_transactions.csv"))
test_df.to_csv(os.path.join("..", "data", "test_transactions.csv"))

user_df = pd.read_csv(os.path.join("..", "data", "sd254_users.csv"))

train_users_df = user_df.iloc[train_users]
test_users_df = user_df.iloc[test_users]

train_users_df.to_csv(os.path.join("..", "data", "train_users.csv"))
test_users_df.to_csv(os.path.join("..", "data", "test_users.csv"))
