from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from utils.preprocessing import Preprocessing
import numpy as np

class CreditDataset(Dataset):
    def __init__(self, transaction_filename: str, demographic_filename: str, num_weeks: int = 6):
        self.preprocessor = Preprocessing(-1)
        demographic_df = pd.read_csv(demographic_filename)
        self.demographic_data = self.load_demographic_dataset(demographic_df)

        self.num_weeks = num_weeks
        transaction_df = pd.read_csv(transaction_filename)
        self.transaction_data = self.load_transaction_dataset(transaction_df)
        self.num_users = len(transaction_df["User"].unique())
        self.total_transactions = sum(len(user_data) for user_data in self.transaction_data)
    
    def load_demographic_dataset(self, demographic_df: pd.DataFrame):
        demographic_df = self.preprocessor.prepare_dataset(demographic_df)
        tensor_data = []
        for _, row in demographic_df.iterrows():
            tensor_data.append(
                self.preprocessor.preprocess_transaction(row.to_dict()))
        
        tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
        return tensor_data

    def load_transaction_dataset(self, transaction_df: pd.DataFrame):

        transaction_df['Date'] = pd.to_datetime(transaction_df[['Year', 'Month', 'Day']])
        transaction_df = transaction_df.sort_values(by=['User', 'Date'])

        tensor_data = []

        for _, user_transactions in transaction_df.groupby("User"):
            user_transactions = user_transactions.sort_values("Date")
            user_transaction_data = []
            start_index = 0
            while start_index < len(user_transactions):
                start_date = user_transactions.iloc[start_index]["Date"]
                end_date = start_date + pd.Timedelta(weeks=self.num_weeks)

                subset = user_transactions[
                    (user_transactions["Date"] >= start_date) & 
                    (user_transactions["Date"] < end_date)
                ]
                transactions = subset.to_dict(orient="records")

                aggregated_transaction = self.preprocessor.preprocess_transaction(transactions)
                user_transaction_data.append(aggregated_transaction)

                start_index += len(subset)
            
            tensor_data.append(user_transaction_data)

        tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
        
        return tensor_data
    
    def __getitem__(self, index: int):
        
        user_idx = np.random.randint(0, self.num_users)
        demographic_data = self.demographic_data[user_idx]
        transaction_idx = torch.randint(0, self.transaction_data[user_idx].shape[0], (1,)).item()
        transaction_data = self.transaction_data[user_idx][transaction_idx]

        return [demographic_data, transaction_data]
    
    def __len__(self):
        return self.total_transactions