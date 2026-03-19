# Applied-active-inference
Warehouse experiments with Kaggle datasets

## Setup

```bash
pip install -r requirements.txt
```

## Downloading Datasets

This project uses four Kaggle datasets. To download them all:

```bash
python download_datasets.py
```

You need Kaggle credentials. Either:
1. Place your `kaggle.json` in `~/.kaggle/` (download from https://www.kaggle.com/settings → API → Create New Token)
2. Or set environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`

## Target
We want to minimize the amount of held stock for items based on uncertainty and rate of daily sales.
Based on the probability distribution of daily sales, we determine a stock level within 5 standard deviations of daily.
Given this, and the uncertainty in receiving a new shipment from each supplier, we then determine each day
whether or not to request a new shipment, and from which supplier, for each product.

It is assumed that shipments can only be received from suppliers already seen in the dataset.

### Traditional statistics approach
Probability distribution is determined by best-fit from a collection of standard probability models. Reliability of
supplier is determined by simple probability analysis of delay.

### Datasets
## Target Dataset
- [Inventory Management (Grocery)](https://www.kaggle.com/datasets/mustofaahmad/inventory-management-grocery-industry)

## Obsolete Dataset
- [Logistics and Supply Chain](https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset)
- [Car Sales Report](https://www.kaggle.com/datasets/missionjee/car-sales-report)
- [Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
