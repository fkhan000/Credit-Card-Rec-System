
# 📈 Credit Card Recommendation System

A recommendation system for credit card transactions built using machine learning.

## 📂 Dataset

To train the model, make sure to download the dataset from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions) and place the CSV files in the `data` folder.  
> **Note:** The `sd254_cards.csv` file should already be present in the `data` folder. 

## 📦 Installation

Install the required packages listed in the `requirements.txt` file by running the following command in the `src` directory:  
```
pip install -r requirements.txt
```

## 🔄 Data Preparation

Split the dataset into training and testing sets by running:  
```
python src/train_test_split.py
```

## 🚀 Training the Model

To train the model, navigate to the `src` folder and run:  
```
python -m card_rec.train
```
