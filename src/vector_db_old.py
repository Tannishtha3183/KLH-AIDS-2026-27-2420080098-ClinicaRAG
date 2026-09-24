"""
vector_db.py

Member 1
ClinicaRAG Project

Responsibilities
----------------
- Read MedQuAD dataset
- Parse XML files
- Chunk medical knowledge
- Generate embeddings
- Store embeddings in ChromaDB
- Query the medical knowledge base
"""
import xml.dom.minidom

from pathlib import Path
from glob import glob
import xml.etree.ElementTree as ET

import chromadb


# -----------------------------
# Project Paths
# -----------------------------

MEDQUAD_PATH = Path("data/raw/medquad")

VECTOR_DB_PATH = Path("data/vector_store")


def verify_dataset() -> None:
    """
    Verifies that the MedQuAD dataset exists.
    """

    if not MEDQUAD_PATH.exists():
        raise FileNotFoundError(
            f"MedQuAD dataset not found: {MEDQUAD_PATH}"
        )

    print("MedQuAD dataset found.")
    print(f"Location: {MEDQUAD_PATH}")

def find_xml_files() -> list[Path]:
    """
    Finds all XML files inside the MedQuAD dataset.
    """

    xml_files = [
        Path(file)
        for file in glob(
            str(MEDQUAD_PATH / "**" / "*.xml"),
            recursive=True,
        )
    ]

    print(f"\nFound {len(xml_files)} XML files.")

    return xml_files

def inspect_xml_file(xml_file: Path) -> None:
    """
    Reads one MedQuAD XML file and prints its structure.
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    print("\nRoot Tag:")
    print(root.tag)

    print("\nImmediate Child Tags:")

    for child in root:
        print(child.tag)

def extract_qa_pairs(xml_file: Path) -> list[dict]:
    """
    Extracts Question-Answer pairs from a MedQuAD XML file.

    Parameters
    ----------
    xml_file : Path
        Path to a MedQuAD XML file.

    Returns
    -------
    list[dict]
        List of question-answer dictionaries.
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    qa_pairs = []

    qa_section = root.find("QAPairs")

    if qa_section is None:
        return qa_pairs

    for pair in qa_section.findall("QAPair"):

        question = pair.findtext("Question")

        answer = pair.findtext("Answer")

        qa_pairs.append(
            {
                "question": question.strip() if question else "",
                "answer": answer.strip() if answer else "",
            }
        )

    return qa_pairs

def inspect_first_qapair(xml_file: Path) -> None:
    """
    Prints the first QAPair XML exactly as it appears.
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    qa_section = root.find("QAPairs")

    if qa_section is None:
        return

    first_pair = qa_section.find("QAPair")

    if first_pair is None:
        return

    xml_string = ET.tostring(
        first_pair,
        encoding="unicode"
    )

    pretty_xml = xml.dom.minidom.parseString(
        xml_string
    ).toprettyxml()

    print(pretty_xml)


# <<< INSERT THIS FUNCTION HERE >>>

def inspect_complete_xml(xml_file: Path) -> None:
    """
    Prints the complete XML document.
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    xml_string = ET.tostring(
        root,
        encoding="unicode"
    )

    pretty_xml = xml.dom.minidom.parseString(
        xml_string
    ).toprettyxml()

    print(pretty_xml)


# <<< AFTER THAT COMES __main__ >>>

if __name__ == "__main__":
    print("=" * 60)
    print("Vector Database Module")
    print("=" * 60)

    verify_dataset()

    xml_files = find_xml_files()

    print("\nFirst 5 XML files:\n")

    for file in xml_files[:5]:
        print(file)
    print("\n" + "=" * 60)
    print("Inspecting First XML File")
    print("=" * 60)

    inspect_xml_file(xml_files[0])

    print("\n" + "=" * 60)
    print("Extracting Question-Answer Pairs")
    print("=" * 60)

    qa_pairs = extract_qa_pairs(xml_files[0])

    print(f"Total QA Pairs: {len(qa_pairs)}")

    print("\nFirst QA Pair:\n")

    print(qa_pairs[0])

    print("\n" + "=" * 60)
    print("First QAPair XML")
    print("=" * 60)

    inspect_first_qapair(xml_files[0])

    print("\n" + "=" * 60)
    print("Complete XML Document")
    print("=" * 60)

    inspect_complete_xml(xml_files[0])

