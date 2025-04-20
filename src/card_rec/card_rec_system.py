"""
This module defines a recommendation system using neural collaborative filtering. 

It includes:
- `TransactionModel`: Processes user transaction history.
- `DemographicModel`: Processes user demographic information.
- `UserLatentNet`: Combines transaction and demographic embeddings into a user latent vector.
- `RecommendationSystem`: Uses user and item embeddings to compute a relevance score.

The system is designed for recommending credit cards to users based on their transaction and demographic data.
(see docs/Rec_System_Model_Architecture for visualization of architecture)
"""
import torch
from torch import nn, Tensor
from typing import Tuple
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr


class TransactionModel(nn.Module):
    """Component of user latent model for processing user transaction history."""
    def __init__(self, transaction_dim: int, latent_dim: int):
        super(TransactionModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=transaction_dim, nhead=1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.output_layer = nn.Linear(transaction_dim, latent_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        ut = self.encoder(x)
        ut = self.output_layer(ut)
        ut = ut.mean(dim=1)
        return ut

class DemographicModel(nn.Module):
    """Component of demographic model for processing user demographic information"""
    def __init__(self, demographic_dim: int, latent_dim: int):
        super(DemographicModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(demographic_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)   

class UserLatentNet(nn.Module):
    """User latent model that produes user latent vector in recommendation system"""
    def __init__(self, transaction_dim: int, demographic_dim: int, latent_dim: int):
        super(UserLatentNet, self).__init__()
        self.transactionModel = TransactionModel(transaction_dim, latent_dim)
        self.demographicModel = DemographicModel(demographic_dim, latent_dim)
        self.latentModel = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        ud = self.demographicModel(x[0])
        ut = self.transactionModel(x[1])
        #print("Demographic output: ", ud)
        #print("Transaction output: ", ut)
        ul = self.latentModel(torch.concat((ud, ut), dim=-1))
        return ul

class RecommendationSystem(nn.Module):
    """Recommendation system model that uses neural collaborative filtering to produce score between a user and item (credit card)"""
    def __init__(self, 
                 transaction_dim: int,
                 demographic_dim: int,
                 user_latent_dim: int,
                 num_items: int,
                 item_latent_dim: int,
                 num_MF_layers: int,
                 MF_hidden_size: int):
        super(RecommendationSystem, self).__init__()
        self.userLatentModel = UserLatentNet(transaction_dim, demographic_dim, user_latent_dim)
        # we store the item latents in an embedding matrix
        self.itemEmbeddings = nn.Embedding(num_items, item_latent_dim)

        # we then take the user and item vector, concatenate them, and feed them into an MLP
        # to compute a score from 0 to 1 indicating how much of a fit the credit card is for the user
        self.nonLinearMF = nn.Sequential(
            nn.Linear(user_latent_dim + item_latent_dim, MF_hidden_size),
            *[nn.Sequential(nn.Linear(MF_hidden_size, MF_hidden_size), nn.ReLU()) for _ in range(num_MF_layers)]
        )
        self.NCF = nn.Sequential(
            self.nonLinearMF,
            nn.Linear(MF_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tuple[Tuple[Tensor, Tensor], Tensor]) -> Tensor:
        user_latent = self.userLatentModel(x[0])
        item_indices = x[1].squeeze(-1)
        item_latent = self.itemEmbeddings(item_indices)
        score = self.NCF(torch.concat((user_latent, item_latent), dim=-1))
        return score
    
    def learn(self, train_data, optimizer, num_epochs):
        loss_fn = torch.nn.MSELoss()
        
        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            loss = 0
            iterations = 1
            pbar = tqdm(train_data, desc="Loss: 0.0000")
            for batch in pbar:
                user_input, item_input, label = batch
                predicted_score = self((user_input, item_input))
                
                optimizer.zero_grad()
                loss = loss_fn(predicted_score.squeeze(-1), label.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if iterations % 100 == 0:
                    current_loss = total_loss / (iterations * len(user_input))
                    pbar.set_description(f"Loss: {2*current_loss:.4f}")
                iterations += 1


            print(f"Epoch {epoch}/{num_epochs}, Average Loss: {total_loss / len(train_data):.4f}")
    
    
    def predict(self, data):
        self.eval()
        total_loss = 0.0
        total_samples = 0
        loss_fn = torch.nn.MSELoss(reduction='sum')

        with torch.no_grad():
            for _ in range(5):
                for batch in data:
                    user_input, item_input, label = batch
                    predicted_score = self((user_input, item_input)).squeeze(-1)

                    loss = loss_fn(predicted_score, label.float())
                    total_loss += loss.item()
                    total_samples += label.size(0)

        mse = total_loss / total_samples if total_samples > 0 else float('inf')
        return mse
    
    def rank_cards(self, user_data, card_indices):
        demo, tx = user_data

        if demo.dim() == 1:
            demo = demo.unsqueeze(0)
        if tx.dim() == 2:
            tx = tx.unsqueeze(0)
        
        scores = []
        for card_index in card_indices:
            score = self(((demo, tx), torch.tensor([[card_index]]))).item()
            scores.append(score)
        return scores


    def evaluate_ranking(self, data, num_trials=15):

        self.eval()
        total_corr = 0
        count = 0
        with torch.no_grad():
            for index in range(len(data)):
                user_corr = 0
                for _ in range(num_trials):
                    dp = data[index]
                    card_indices = dp[1]

                    if len(card_indices) < 2:
                        continue

                    actual_ratings = dp[2]
                    predicted_ratings = self.rank_cards(dp[0], card_indices)

                    corr = spearmanr(actual_ratings, predicted_ratings).correlation
                    user_corr += corr
                if len(card_indices) < 2:
                    continue
                total_corr += (user_corr / num_trials)
                count += 1
        
        return total_corr/count
    
    def recall_at_k(self, data, k=3):
        self.eval()
        total_recall = 0.0
        valid_users = 0

        with torch.no_grad():
            for index in range(len(data)):
                user_data, card_indices, actual_ratings = data[index]

                if len(card_indices) < 2:
                    continue

                predicted_scores = self.rank_cards(user_data, card_indices)

                k_eff = min(k, len(predicted_scores))

                top_k_pred_indices = torch.topk(torch.tensor(predicted_scores), k_eff).indices.tolist()

                top_k_actual_indices = torch.topk(actual_ratings, k_eff).indices.tolist()

                hits = len(set(top_k_pred_indices) & set(top_k_actual_indices))
                recall = hits / k_eff

                total_recall += recall
                valid_users += 1

        return total_recall / valid_users if valid_users > 0 else float("nan")
