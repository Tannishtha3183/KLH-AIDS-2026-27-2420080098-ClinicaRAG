# src/clinical_bert.py
from transformers import pipeline
import string

print("Initializing Biomedical NER model...")

ner_pipeline = pipeline(
    "ner", 
    model="d4data/biomedical-ner-all", 
    aggregation_strategy="simple"
)

def expand_boundaries(text: str, start: int, end: int):
    """Expands character indices outward to capture the whole word."""
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end

def extract_clinical_entities(cleaned_note: str) -> dict:
    """
    Analyzes a clinical note and extracts medical entities mapped to a standard JSON schema.
    Uses boundary expansion to prevent fragmented tokens.
    """
    raw_entities = ner_pipeline(cleaned_note)
    
    structured_output = {
        "symptoms": [],
        "suspected_diagnoses": [],
        "medications": [],
        "lab_tests_mentioned": []
    }
    
    label_map = {
        "Sign_symptom": "symptoms",
        "Biological_structure": "symptoms",
        "Disease_syndrome": "suspected_diagnoses",
        "Medication": "medications",
        "Clinical_drug": "medications",
        "Diagnostic_procedure": "lab_tests_mentioned",
        "Laboratory_procedure": "lab_tests_mentioned",
        "Therapeutic_procedure": "lab_tests_mentioned"
    }

    valid_entities = [e for e in raw_entities if e['entity_group'] in label_map]
    valid_entities = sorted(valid_entities, key=lambda x: x['start'])

    if not valid_entities:
        return structured_output

    first_start, first_end = expand_boundaries(cleaned_note, valid_entities[0]['start'], valid_entities[0]['end'])
    current_type = label_map[valid_entities[0]['entity_group']]
    current_start = first_start
    current_end = first_end

    for ent in valid_entities[1:]:
        target_list = label_map[ent['entity_group']]
        exp_start, exp_end = expand_boundaries(cleaned_note, ent['start'], ent['end'])

        if target_list == current_type and exp_start <= current_end + 1:
            current_end = max(current_end, exp_end)
        else:
            extracted_text = cleaned_note[current_start:current_end].strip()
            if extracted_text:
                structured_output[current_type].append(extracted_text)
            
            current_type = target_list
            current_start = exp_start
            current_end = exp_end

    extracted_text = cleaned_note[current_start:current_end].strip()
    if extracted_text:
        structured_output[current_type].append(extracted_text)

    for key in structured_output:
        structured_output[key] = list(set(structured_output[key]))

    return structured_output

def evaluate_extractor(test_data: list) -> dict:
    """
    Evaluates the NER extractor against a set of ground truth test cases.
    Calculates Precision, Recall, and F1-Score.
    """
    tp = 0
    fp = 0
    fn = 0

    for item in test_data:
        note = item["note"]
        ground_truth = item["expected"]
        predictions = extract_clinical_entities(note)

        # Flatten to sets of (category, lowercase_entity) for exact matching
        gt_set = set((cat, val.lower()) for cat, vals in ground_truth.items() for val in vals)
        pred_set = set((cat, val.lower()) for cat, vals in predictions.items() for val in vals)

        tp += len(gt_set.intersection(pred_set))
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1_Score": round(f1_score, 2)
    }

if __name__ == "__main__":
    import json
    
    # We define a small mock evaluation dataset with Ground Truth expected outputs
    mock_eval_data = [
        {
            "note": "Patient presents with severe chest pain and dyspnea. Prescribed aspirin and ordered an ECG.",
            "expected": {
                "symptoms": ["chest pain", "dyspnea"],
                "suspected_diagnoses": [],
                "medications": ["aspirin"],
                "lab_tests_mentioned": ["ecg"]
            }
        },
        {
            "note": "Diagnosed with acute myocardial infarction. Started on lisinopril.",
            "expected": {
                "symptoms": [],
                "suspected_diagnoses": ["acute myocardial infarction"],
                "medications": ["lisinopril"],
                "lab_tests_mentioned": []
            }
        }
    ]
    
    print("\n--- Running NER Evaluation Pipeline ---")
    metrics = evaluate_extractor(mock_eval_data)
    print(json.dumps(metrics, indent=4))