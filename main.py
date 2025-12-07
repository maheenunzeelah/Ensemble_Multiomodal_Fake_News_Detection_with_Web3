#%%

from helpers.data_utils import (
    load_and_combine,
    keep_selected_columns,
    subsample_rows
)
from helpers.ner_utils import extract_person_entities, filter_entities_by_frequency, flatten_entities, normalize_entity, save_json
from collections import Counter
import pandas as pd
import ast

#%%
# 1. Load & combine
df = load_and_combine(
    "datasets/raw/multimodal_train.tsv",
    "datasets/raw/multimodal_validate.tsv",
    "datasets/raw/multimodal_test_public.tsv"
)

# 2. Keep required columns
df = keep_selected_columns(df)

# 3. Subsample 12,000 rows
df = subsample_rows(df, n=12000)

df.to_csv("datasets/processed/fakereddit.csv", index=False)
print("Interim dataset saved as fakereddit.csv")
# 4. Apply entity extraction
df["entities"] = df["clean_title"].apply(extract_person_entities)
df.info()
# 5. Save output
df.to_csv("datasets/processed/final_dataset_with_entities.csv", index=False)

print("Dataset processing completed! Saved as final_dataset_with_entities.csv")


# %%
df = pd.read_csv("./datasets/processed/final_dataset_with_entities.csv")

# Convert entities from string to list
df['entities'] = df['entities'].apply(ast.literal_eval)

# Flatten all entities
all_entities = [entity for row in df['entities'] for entity in row]

# Count occurrences
entity_counts = Counter(all_entities)
total_entities = len(all_entities)
print("Total number of entities (all occurrences):", total_entities)
# Print entities and their counts
for entity, count in entity_counts.most_common():
    print(f"{entity}: {count}")
# %%
alias_map = {
    "trump": "donald trump",
    "trumps": "donald trump",
    "donald trump": "donald trump",
    "donald trumps": "donald trump",
    
    "hitler": "adolf hitler",
    "adolf hitler": "adolf hitler",

    "obama": "barack obama",
    "barack obama": "barack obama",

    "putin": "vladimir putin",
    "vladimir putin": "vladimir putin",

    "kim jong un": "kim jong-un",
    "kim jongun": "kim jong-un",
}

# Flatten list of entities
all_entities = flatten_entities(df['entities'])

# Normalize each entity
normalized_entities = [
    normalize_entity(ent, alias_map) 
    for ent in all_entities
]

# Count occurrences
entity_counts = Counter(normalized_entities)

# Filter entities with occurrences > 5
entities_over_5 = filter_entities_by_frequency(entity_counts, min_count=5)

# Unique entity list
unique_entities = sorted(entity_counts.keys())

# Save to JSON
save_json(unique_entities, "./datasets/processed/unique_entities.json")
# %%
