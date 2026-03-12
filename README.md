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

### Datasets
- [Logistics and Supply Chain](https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset)
- [Car Sales Report](https://www.kaggle.com/datasets/missionjee/car-sales-report)
- [Inventory Management (Grocery)](https://www.kaggle.com/datasets/mustofaahmad/inventory-management-grocery-industry)
- [Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
