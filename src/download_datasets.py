from datasets import load_dataset
from pathlib import Path
import pandas as pd


def download_pubmedqa():
    """
    Downloads the PubMedQA dataset and saves the train split as CSV.
    """

    print("Downloading PubMedQA...")

    dataset = load_dataset("pubmed_qa", "pqa_labeled")

    train = dataset["train"]

    df = pd.DataFrame(train)

    output_dir = Path("data/raw/pubmedqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pubmedqa_train.csv"

    df.to_csv(output_file, index=False)

    print(f"Dataset saved to: {output_file}")


if __name__ == "__main__":
    download_pubmedqa()