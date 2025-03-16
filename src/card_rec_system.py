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

class TransactionModel(nn.Module):
    """Component of user latent model for processing user transaction history."""
    def __init__(self, transaction_dim: int, latent_dim: int):
        super(TransactionModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=transaction_dim, nhead=2)
        self.encoder = nn.TransformerEncoder(encoder_layer, 3)
        self.output_layer = nn.Linear(transaction_dim, latent_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        ut = self.encoder(x)
        ut = self.output_layer(ut)
        return ut

class DemographicModel(nn.Module):
    """Component of demographic model for processing user demographic information"""
    def __init__(self, demographic_dim: int, latent_dim: int):
        super(DemographicModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(demographic_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
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
            nn.Linear(2 * latent_dim, latent_dim)
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
        #print("user_latent: ", user_latent)
        item_indices = x[1].squeeze(-1)
        item_latent = self.itemEmbeddings(item_indices)
        #print("item_latent: ", item_latent)
        score = self.NCF(torch.concat((user_latent, item_latent), dim=-1))
        return score
    
    def learn(self, train_data, optimizer, num_epochs):
        loss_fn = torch.nn.BCELoss()
        
        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            
            for batch in tqdm(train_data):
                user_input, item_input, label = batch
                #print("Demographic Input: ", user_input[0])
                predicted_score = self((user_input, item_input))
                
                optimizer.zero_grad()
                loss = loss_fn(predicted_score.squeeze(-1), label.float())
                print(loss)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            print(f"Epoch {epoch}/{num_epochs}, Average Loss: {total_loss / len(train_data):.4f}")