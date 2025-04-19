from .load_data import load_data
from .card_rec_system import RecommendationSystem
import torch.optim as optim
import os
import torch

if __name__ == "__main__":
    training_data, train_dataloader, test_data, test_dataloader, test_rank_dataloader = load_data(num_weeks=6)
    model = RecommendationSystem(8, 7, 100, test_data.num_credit_cards, 256, 4, 256)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)

    model.learn(train_dataloader, optimizer, 1)
    model.eval()

    #print("MSE Loss on Test Set: ",
    #      model.predict(test_dataloader))
    print("Average Correlation Between Predicted and True Ranking",
          model.evaluate_ranking(test_rank_dataloader))

    torch.save(model,
               os.path.join("..", "models", "rec_system.pth"))