# helpers/ner_utils.py
import spacy
import re
import json

spacy.prefer_gpu()  # Use GPU if available

# load model once
nlp = spacy.load("en_core_web_trf")

def extract_entities(text):
    """Extract all named entities: [(text, label), ...]"""
    doc = nlp(str(text))
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_person_entities(text):
    """Extract only person names"""
    doc = nlp(str(text))
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, and trim whitespace.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def spacy_lemmatize(text: str) -> str:
    """
    Lemmatize using spaCy (better than NLTK).
    Handles multi-word entities.
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def apply_alias_mapping(entity: str, alias_map: dict) -> str:
    """
    Map entity to a canonical name if found in alias_map.
    Otherwise return the entity unchanged.
    """
    return alias_map.get(entity, entity)

def normalize_entity(entity: str, alias_map: dict) -> str:
    """
    Full normalization:
    - Clean
    - Lemmatize
    - Alias mapping
    """
    entity = clean_text(entity)
    entity = spacy_lemmatize(entity)
    entity = apply_alias_mapping(entity, alias_map)
    return entity


def flatten_entities(entity_list_col):
    """
    Flatten a column containing lists of entities.
    """
    return [entity for row in entity_list_col for entity in row]


def save_json(data, path: str):
    """
    Save Python data to JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def filter_entities_by_frequency(entity_counts, min_count=5):
    """
    Return a list of entities that occur at least `min_count` times.
    entity_counts should be a Counter object.
    """
    return [entity for entity, count in entity_counts.items() if count > min_count]
