from typing import Dict, Any, List
from torch import Tensor
from datetime import datetime
import torch
class Preprocessing:
    def __init__(self,
                 mcc_default,
                 transaction_means: Tensor | None = None,
                 transaction_vars: Tensor | None = None,
                 demographic_means: Tensor | None = None,
                 demographic_vars: Tensor | None = None):
        self.default_agg_transaction = {"MCC_1": mcc_default,
                                        "MCC_2": mcc_default,
                                        "MCC_1_Amount": 0,
                                        "MCC_2_Amount": 0,
                                        "num_transactions": 0,
                                        "Average_Zip": -1,
                                        "Max_Zip": -float("inf"),
                                        "Min_Zip": float("inf")}
        self.transaction_fields = ["MCC_1", "MCC_2", "MCC_1_Amount", "MCC_2_Amount", "num_transactions", "Average_Zip", "Max_Zip", "Min_Zip"]
        
        self.demographic_fields = ["Birth Year",
                                   "Gender",
                                   "Zipcode",
                                   "Per Capita Income - Zipcode",
                                   "Yearly Income - Person",
                                   "Total Debt",
                                   "FICO Score"]
        self.transaction_means = transaction_means
        self.transaction_vars = transaction_vars
        self.demographic_means = demographic_means
        self.demographic_vars = demographic_vars

    
    def get_date(self, year, month, day) -> datetime:
        return datetime(year, month, day)

    def aggregate_transaction(self, aggregated_transaction, mcc_amounts: Dict[str, int]):
        mccs = sorted(mcc_amounts.keys(), key=mcc_amounts.get, reverse=True)
        aggregated_transaction["MCC_1"] = mccs[0]
        aggregated_transaction["MCC_2"] = mccs[1]
        aggregated_transaction["MCC_1_Amount"] = mcc_amounts[mccs[0]]
        aggregated_transaction["MCC_2_Amount"] = mcc_amounts[mccs[1]]
        aggregated_transaction["Average_Zip"] /= aggregated_transaction["num_transactions"]

        return aggregated_transaction

    def preprocess_transaction(self, transactions: List[Dict[str, Any]]) -> Tensor:
        """
            Preprocesses the transaction history so it can be fed into the transaction model. 
            (Aggregating transactions, Binning the merchants, Obtaining counts by merchant category, etc.)
            and returns an TxF tensor where T is the number of time periods (maybe weeks?) and F is the number of features

            --- Args
            transactions List[Dict[str, Any]]: A list of transactions within a 6 week time period sorted from most recent to least
        """
        weekly_transactions: List[Dict[str, float]] = []
        aggregated_transaction: Dict[str, float] = self.default_agg_transaction
        mcc_amounts = {}
        start_date = self.get_date(transactions[0]["Year"],
                                   transactions[0]["Month"],
                                   transactions[0]["Day"])
        
        for transaction in transactions:
            transaction_date = self.get_date(transaction["Year"], transaction["Month"], transaction["Day"])
            if abs((transaction_date - start_date).days) > 7:
                aggregated_transaction = self.aggregate_transaction(aggregated_transaction, mcc_amounts)
                weekly_transactions.append(aggregated_transaction)
                aggregated_transaction = self.default_agg_transaction
                mcc_amounts = {}
                start_date = self.get_date(transactions[0]["Year"],transactions[0]["Month"],transactions[0]["Day"])
            
            mcc_amounts[transaction["MCC"]] = mcc_amounts.get(transaction["MCC"], 0) + transaction["Amount"]
            aggregated_transaction["Average_Zip"] += transaction["Zip"]
            
        transaction_data = []
        for aggregate in weekly_transactions:
            transaction_data.append(list(map(aggregate.get, self.transaction_fields)))
        
        tensor_data = Tensor(transaction_data, dtype=torch.float32)
        if self.transaction_vars:
            tensor_data = torch.div((tensor_data - self.transaction_means),
                                    self.transaction_vars)
        return tensor_data
    

    def preprocess_demographic(self, demographic_info: Dict[str, Any]) -> Tensor:
        """
        Preprocesses the demographic information provided in data/sd254_users.csv (credit limit in sd254_cards.csv could be useful too).
        Returns a tensor of size D where D is the number of demographic features.
        """

        gender = demographic_info["Gender"]
        demographic_info["Gender"] = 1 if gender =="Male" else 0

        demographic_data = Tensor(list(map(demographic_info.get, self.demographic_fields)), dtype=torch.float32)

        if self.demographic_means:
            demographic_data = torch.div((demographic_data - self.demographic_means),
                                    self.demographic_vars)
        return demographic_data
