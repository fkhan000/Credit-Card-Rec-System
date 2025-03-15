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

class TransactionModel(nn.Module):
    """Component of user latent model for processing user transaction history."""
    def __init__(self, transaction_dim: int, latent_dim: int):
        super(TransactionModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=transaction_dim, nhead=3)
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
            nn.Linear(demographic_dim, latent_dim)
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
        item_latent = self.itemEmbeddings(x[1])
        score = self.NCF(torch.concat((user_latent, item_latent), dim=-1))
        return score
