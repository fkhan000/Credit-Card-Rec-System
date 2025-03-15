# Preprocessing Class Documentation

## Overview
The `Preprocessing` class is designed to handle data preprocessing for both transaction and demographic data, making it suitable for input into machine learning models. It includes methods for data normalization, transaction aggregation, and demographic processing.

## Class: `Preprocessing`

### Attributes:
- `default_agg_transaction` (Dict[str, Any]): Default values for aggregated transaction data.
- `transaction_fields` (List[str]): List of transaction-related feature names.
- `demographic_fields` (List[str]): List of demographic feature names.
- `transaction_means` (Tensor, optional): Mean values for transaction normalization.
- `transaction_vars` (Tensor, optional): Variance values for transaction normalization.
- `demographic_means` (Tensor, optional): Mean values for demographic normalization.
- `demographic_vars` (Tensor, optional): Variance values for demographic normalization.

## Initialization:
```python
Preprocessing(mcc_default, transaction_means=None, transaction_vars=None, demographic_means=None, demographic_vars=None)
```
#### Parameters:
- `mcc_default`: Default value for missing MCC codes.
- `transaction_means` (Tensor, optional): Mean values for transaction normalization.
- `transaction_vars` (Tensor, optional): Variance values for transaction normalization.
- `demographic_means` (Tensor, optional): Mean values for demographic normalization.
- `demographic_vars` (Tensor, optional): Variance values for demographic normalization.

## Methods:

### `get_date(year: int, month: int, day: int) -> datetime`
Returns a `datetime` object for the given year, month, and day.

#### Parameters:
- `year` (int): Year component.
- `month` (int): Month component.
- `day` (int): Day component.

#### Returns:
- `datetime`: A `datetime` object representing the specified date.

### `aggregate_transaction(aggregated_transaction: Dict[str, Any], mcc_amounts: Dict[str, int]) -> Dict[str, Any]`
Aggregates transaction data based on merchant category codes (MCC).

#### Parameters:
- `aggregated_transaction` (Dict[str, Any]): A dictionary containing aggregated transaction values.
- `mcc_amounts` (Dict[str, int]): A dictionary mapping MCC codes to total transaction amounts.

#### Returns:
- `Dict[str, Any]`: The updated aggregated transaction data.

### `preprocess_transaction(transactions: List[Dict[str, Any]]) -> Tensor`
Preprocesses a list of transactions for use in a machine learning model.

This involves aggregating transactions, binning merchants, and computing counts by merchant category.

#### Parameters:
- `transactions` (List[Dict[str, Any]]): A list of transaction dictionaries within a six-week time period, sorted from most recent to least recent.

#### Returns:
- `Tensor`: A PyTorch tensor representing the processed transaction data.

### `preprocess_demographic(demographic_info: Dict[str, Any]) -> Tensor`
Preprocesses demographic data for input into a machine learning model.

The function maps categorical data to numerical values and normalizes the data if required.

#### Parameters:
- `demographic_info` (Dict[str, Any]): A dictionary containing demographic information.

#### Returns:
- `Tensor`: A PyTorch tensor representing the processed demographic data.

## Usage Example:
```python
from torch import tensor

# Example initialization
preprocessor = Preprocessing(mcc_default=0, transaction_means=tensor([1.0]), transaction_vars=tensor([1.0]))

# Example transaction processing
transactions = [
    {"Year": 2023, "Month": 5, "Day": 20, "MCC": "5411", "Amount": 50, "Zip": 10001},
    {"Year": 2023, "Month": 5, "Day": 21, "MCC": "5812", "Amount": 30, "Zip": 10001},
]
processed_transactions = preprocessor.preprocess_transaction(transactions)

# Example demographic processing
demographics = {"Gender": "Male", "Birth Year": 1990, "Zipcode": 10001, "Per Capita Income - Zipcode": 55000, "Yearly Income - Person": 75000, "Total Debt": 10000, "FICO Score": 700}
processed_demographics = preprocessor.preprocess_demographic(demographics)
```

## Conclusion
The `Preprocessing` class simplifies data preparation by normalizing features and aggregating transactions. This ensures that the processed data is optimized for machine learning applications.
