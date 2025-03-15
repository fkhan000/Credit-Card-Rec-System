class Preprocessing:
    def preprocess_transaction(self, x):
        """
            Preprocesses the transaction history so it can be fed into the transaction model. 
            (Aggregating transactions, Binning the merchants, Obtaining counts by merchant category, etc.)
            and returns an TxF tensor where T is the number of time periods (maybe weeks?) and F is the number of features
        """
        #TODO: Add preprocessing step for transaction history
        pass

    def preprocess_demographic(self, x):
        """
        Preprocesses the demographic information provided in data/sd254_users.csv (credit limit in sd254_cards.csv could be useful too).
        Returns a tensor of size D where D is the number of demographic features.
        """
        #TODO: Add preprocessing step for demographic information
        pass 
