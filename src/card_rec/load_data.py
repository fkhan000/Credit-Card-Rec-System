from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from utils.preprocessing import Preprocessing
import numpy as np
import os
from tqdm import tqdm
class CreditDataset(Dataset):
    def __init__(self,
                 transaction_filename: str,
                 demographic_filename: str,
                 credit_card_filename: str,
                 num_weeks: int = 6):

        self.num_weeks = num_weeks
        transaction_df = pd.read_csv(transaction_filename,
                                     usecols=["User", "Merchant Category", "Year", "Month", "Day", "Zip", "Amount"])
        self.preprocessor = Preprocessing(transaction_df)
        self.transaction_data = self.load_transaction_dataset(transaction_df)
        self.num_users = len(transaction_df["User"].unique())
        self.total_transactions = sum(len(user_data) for user_data in self.transaction_data)

        demographic_df = pd.read_csv(demographic_filename)
        self.demographic_data = self.load_demographic_dataset(demographic_df)
        credit_card_data = pd.read_csv(credit_card_filename)
        self.user_labels = self.load_credit_card_data(credit_card_data)
        self.num_credit_cards = len(self.user_labels[0])

    def load_demographic_dataset(self, demographic_df: pd.DataFrame):
        demographic_df = self.preprocessor.prepare_dataset(demographic_df)
        tensor_data = []
        for _, row in demographic_df.iterrows():
            tensor_data.append(
                self.preprocessor.preprocess_demographic(row.to_dict()))
        
        tensor_data = torch.stack(tensor_data)
        return tensor_data

    def load_transaction_dataset(self, transaction_df: pd.DataFrame):

        transaction_df = self.preprocessor.prepare_dataset(transaction_df)

        transaction_df['Date'] = pd.to_datetime(transaction_df[['Year', 'Month', 'Day']])
        transaction_df = transaction_df.sort_values(by=['User', 'Date'])

        tensor_data = []

        for _, user_transactions in tqdm(transaction_df.groupby("User")):
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
            user_transaction_data = torch.stack(user_transaction_data)
            tensor_data.append(user_transaction_data)

        tensor_data = torch.concatenate(tensor_data)
        
        return tensor_data

    def load_credit_card_data(self, credit_card_df: pd.DataFrame):

        unique_categories = (
            credit_card_df["Card Category"]
            .value_counts()
            .head(10)
            .index.tolist()
        )
        self.card_categories = unique_categories

        user_labels = {}

        category_to_index = {cat: i for i, cat in enumerate(unique_categories)}

        for user_id, group in credit_card_df.groupby("User"):
            rating_vec = np.zeros(len(unique_categories))
            for _, row in group.iterrows():
                cat = row["Card Category"]
                if cat in category_to_index:
                    rating_vec[category_to_index[cat]] = row["Rating"]
            user_labels[user_id] = rating_vec
        
        num_users = max(user_labels.keys()) + 1
        user_label_list = [user_labels.get(i, np.zeros(len(unique_categories))) for i in range(num_users)]

        return user_label_list

        """
        filtered_df = credit_card_df[credit_card_df["Card Category"] != "UNKNOWN"]
        filtered_df["Card Category"] = filtered_df["Card Category"].apply(lambda x: x.split()[1])
        unique_brands = (filtered_df["Card Category"]
                      .value_counts()
                      .head(10)
                      .index.tolist())
        
        # Create one-hot vectors for each user
        user_labels = []
        for user, group in credit_card_df.groupby("User"):
            label = [1 if brand in group["Card Category"].values else 0 for brand in unique_brands]
            user_labels.append(label)
        return user_labels
        """
    
    def __getitem__(self, index: int):
        user_idx = np.random.randint(0, self.num_users)
        demographic_data = self.demographic_data[user_idx]
        transaction_idx = torch.randint(0, self.transaction_data[user_idx].shape[0], (1,)).item()
        transaction_data = self.transaction_data[user_idx][transaction_idx]

        user_ratings = np.array(self.user_labels[user_idx])
        valid_card_indices = np.where(user_ratings > 0)[0]

        if len(valid_card_indices) == 0:
            credit_index = torch.randint(0, self.num_credit_cards, (1,)).item()
            label = 0.0
        else:
            credit_index = np.random.choice(valid_card_indices)
            label = user_ratings[credit_index]

        return [demographic_data, transaction_data], credit_index, torch.tensor(label, dtype=torch.float32)

        """
        target_label = torch.randint(0, 2, (1,)).item()

        user_labels = np.array(self.user_labels[user_idx])
        credit_indices = np.where(user_labels == target_label)[0]
        
        if credit_indices.size == 0:
            credit_index = torch.randint(0, self.num_credit_cards, (1,)).item()
        else:
            credit_index = np.random.choice(credit_indices)
        
        label = self.user_labels[user_idx][credit_index]

        return [demographic_data, transaction_data], credit_index, label
        """
    
    def __len__(self):
        return self.total_transactions

class CreditRankDataset(CreditDataset):
    def __init__(self,
                 transaction_filename: str,
                 demographic_filename: str,
                 credit_card_filename: str,
                 num_weeks: int = 6):
        super().__init__(transaction_filename,
                         demographic_filename,
                         credit_card_filename,
                         num_weeks=num_weeks)
    
    def __getitem__(self, index: int):
        demographic_data = self.demographic_data[index]
        transaction_idx = torch.randint(0, self.transaction_data[index].shape[0], (1,)).item()
        transaction_data = self.transaction_data[index][transaction_idx]

        user_ratings = torch.tensor(self.user_labels[index])
        valid_card_indices = torch.where(user_ratings > 0)[0]

        return [demographic_data, transaction_data], valid_card_indices, user_ratings

    def __len__(self):
        return len(self.user_labels)
        

def load_data(num_weeks: int = 6, batch_size = 64):

    train_demographic_filename = os.path.join("..", "data", "train_users.csv")
    train_transaction_filename = os.path.join("..", "data", "train_transactions.csv")
    credit_card_filename = os.path.join("..", "data", "cc_dataset_with_labels.csv")

    
    training_data = CreditDataset(train_transaction_filename,
                                  train_demographic_filename,
                                  credit_card_filename,
                                  num_weeks=num_weeks)
                                  
    
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True,
    )

    test_demographic_filename = os.path.join("..", "data", "test_users.csv")
    test_transaction_filename = os.path.join("..", "data", "test_transactions.csv")

    test_data = CreditDataset(test_transaction_filename,
                                  test_demographic_filename,
                                  credit_card_filename,
                                  num_weeks=num_weeks)
                                  
    
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True,
    )

    credit_rank_data = CreditRankDataset(test_transaction_filename,
                                         test_demographic_filename,
                                         credit_card_filename,
                                         num_weeks=num_weeks)
    test_rank_dataloader = DataLoader(credit_rank_data)
    return training_data, train_dataloader, test_data, test_dataloader, test_rank_dataloader
