"""Download all Kaggle datasets used in this project via kagglehub."""

import kagglehub

DATASETS = [
    "datasetengineer/logistics-and-supply-chain-dataset",
    "missionjee/car-sales-report",
    "mustofaahmad/inventory-management-grocery-industry",
    "anirudhchauhan/retail-store-inventory-forecasting-dataset",
]

def main():
    for dataset in DATASETS:
        print(f"\nDownloading: {dataset}")
        path = kagglehub.dataset_download(dataset)
        print(f"  -> {path}")

if __name__ == "__main__":
    main()
