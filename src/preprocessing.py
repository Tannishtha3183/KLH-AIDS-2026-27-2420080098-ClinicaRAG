"""
preprocessing.py

Member 1
ClinicalRAG Project

Responsibilities:
- Load clinical notes dataset
- Clean text
- Lowercase conversion
- Stop-word removal
- Lemmatization
- Prepare text for downstream NLP tasks
"""

from pathlib import Path
import re

import pandas as pd
import spacy

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)

# Load SpaCy model only once
nlp = spacy.load("en_core_web_sm")


# Dataset location
DATASET_PATH = Path("data/raw/clinical_notes/mtsamples.csv")


def load_dataset() -> pd.DataFrame:
    """
    Loads the Medical Transcriptions dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset containing clinical notes.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    return df


def clean_text(text: str) -> str:
    """
    Cleans a clinical note using SpaCy.

    Steps:
    1. Lowercase
    2. Remove extra whitespace
    3. Remove numbers
    4. Remove punctuation/special characters
    5. Lemmatize
    6. Remove stop words
    7. Keep alphabetic tokens only

    Parameters
    ----------
    text : str
        Raw clinical note.

    Returns
    -------
    str
        Cleaned and lemmatized text.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
    ]

    cleaned_text = " ".join(tokens)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text preprocessing to the entire dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Dataset with an additional cleaned_text column.
    """

    print("\nPreprocessing clinical notes...")

    processed_df = df.copy()

    processed_df["cleaned_text"] = (
        processed_df["transcription"]
        .fillna("")
        .apply(clean_text)
    )

    return processed_df


def create_bow_matrix(
    df: pd.DataFrame,
    max_features: int = 1000
):
    """
    Creates a Bag-of-Words matrix from the cleaned clinical notes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the 'cleaned_text' column.

    max_features : int
        Maximum vocabulary size.

    Returns
    -------
    tuple
        (vectorizer, bow_matrix)
    """

    if "cleaned_text" not in df.columns:
        raise ValueError(
            "The DataFrame must contain a 'cleaned_text' column."
        )

    vectorizer = CountVectorizer(
        max_features=max_features
    )

    bow_matrix = vectorizer.fit_transform(
        df["cleaned_text"]
    )

    return vectorizer, bow_matrix

def create_tfidf_matrix(
    df: pd.DataFrame,
    max_features: int = 1000
):
    """
    Creates a TF-IDF matrix from the cleaned clinical notes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the 'cleaned_text' column.

    max_features : int
        Maximum vocabulary size.

    Returns
    -------
    tuple
        (vectorizer, tfidf_matrix)
    """

    if "cleaned_text" not in df.columns:
        raise ValueError(
            "The DataFrame must contain a 'cleaned_text' column."
        )

    vectorizer = TfidfVectorizer(
        max_features=max_features
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["cleaned_text"]
    )

    return vectorizer, tfidf_matrix

def save_bow_vocabulary(
    vectorizer: CountVectorizer,
    output_path: Path
) -> None:
    """
    Saves the Bag-of-Words vocabulary to a CSV file.
    """

    vocabulary = pd.DataFrame(
        {
            "term": vectorizer.get_feature_names_out()
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocabulary.to_csv(output_path, index=False)

    print(f"Vocabulary saved to: {output_path}")

def save_tfidf_vocabulary(
    vectorizer: TfidfVectorizer,
    output_path: Path
) -> None:
    """
    Saves the TF-IDF vocabulary to a CSV file.
    """

    vocabulary = pd.DataFrame(
        {
            "term": vectorizer.get_feature_names_out()
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocabulary.to_csv(output_path, index=False)

    print(f"TF-IDF vocabulary saved to: {output_path}")


if __name__ == "__main__":

    df = load_dataset()

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Total Records : {len(df)}")
    print(f"Columns       : {list(df.columns)}")

    sample_note = df["transcription"].iloc[0]

    print("\nRAW NOTE")
    print("-" * 60)
    print(sample_note[:500])

    cleaned = clean_text(sample_note)

    print("\nCLEANED NOTE")
    print("-" * 60)
    print(cleaned[:500])

    # Process the complete dataset
    processed_df = preprocess_dataset(df)

    # Create output directory if it doesn't exist
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "clinical_notes_cleaned.csv"

    processed_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("Dataset preprocessing completed successfully!")
    print("=" * 60)

    print(f"Saved to: {output_path}")

    print(f"\nTotal processed notes: {len(processed_df)}")


    print("\n" + "=" * 60)
    print("Creating Bag-of-Words Matrix...")
    print("=" * 60)

    vectorizer, bow_matrix = create_bow_matrix(processed_df)

    print(f"BoW Matrix Shape : {bow_matrix.shape}")

    print(f"Vocabulary Size  : {len(vectorizer.get_feature_names_out())}")

    print("\nFirst 20 Vocabulary Words:\n")

    print(vectorizer.get_feature_names_out()[:20])

    vocab_path = Path("data/processed/bow_vocabulary.csv")

    save_bow_vocabulary(
        vectorizer,
        vocab_path
    )

    print("\n" + "=" * 60)
    print("Creating TF-IDF Matrix...")
    print("=" * 60)

    tfidf_vectorizer, tfidf_matrix = create_tfidf_matrix(processed_df)

    print(f"TF-IDF Matrix Shape : {tfidf_matrix.shape}")

    print(f"TF-IDF Vocabulary Size : {len(tfidf_vectorizer.get_feature_names_out())}")

    print("\nFirst 20 TF-IDF Vocabulary Words:\n")

    print(tfidf_vectorizer.get_feature_names_out()[:20])

    tfidf_vocab_path = Path("data/processed/tfidf_vocabulary.csv")

    save_tfidf_vocabulary(
        tfidf_vectorizer,
        tfidf_vocab_path
    )


    