from pathlib import Path

from datasets import load_dataset

print("=" * 60)
print("Downloading Complete MedQuAD Dataset")
print("=" * 60)

dataset = load_dataset("prithvi1029/medquad-medical-qa")

output_dir = Path("data/raw/medquad_complete")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "medquad_complete.csv"

dataset["train"].to_csv(output_file, index=False)

print("\nDownload completed successfully!")
print(f"Saved to: {output_file}")

print("\nDataset Information")
print(dataset["train"])