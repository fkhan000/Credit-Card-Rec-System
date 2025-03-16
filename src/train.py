from load_data import load_data
from card_rec_system import RecommendationSystem
import torch.optim as optim
import os
import torch

if __name__ == "__main__":
    training_data, train_dataloader, test_data, test_dataloader = load_data()
    model = RecommendationSystem(8, 7, 300, training_data.num_credit_cards, 100, 3, 128)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    model.learn(train_dataloader, optimizer, 5)

    print("Accuracy: ", model.predict(test_dataloader))

    torch.save(model,
               os.path.join("..", "models", "rec_system.pth"))