"""Download all Kaggle datasets used in this project via kagglehub."""

import kagglehub

DATASETS = [
    "mustofaahmad/inventory-management-grocery-industry",
]

def main():
    for dataset in DATASETS:
        print(f"\nDownloading: {dataset}")
        path = kagglehub.dataset_download(dataset)
        print(f"  -> {path}")

if __name__ == "__main__":
    main()
