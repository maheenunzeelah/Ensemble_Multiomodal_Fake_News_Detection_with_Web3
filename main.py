#%%

from helpers.data_utils import (
    get_image_path,
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
# df = load_and_combine(
#     "datasets/raw/multimodal_train.tsv",
#     "datasets/raw/multimodal_validate.tsv",
#     "datasets/raw/multimodal_test_public.tsv"
# )

# # 2. Keep required columns
# df = keep_selected_columns(df)

# # 3. Subsample 12,000 rows
# df = subsample_rows(df, n=12000)

# df.to_csv("datasets/processed/fakereddit.csv", index=False)
df = pd.read_csv("./datasets/processed/all_data_df_title_entities.csv")
# print("Interim dataset saved as fakereddit.csv")
# 4. Apply entity extraction
df["entities"] = df["clean_title"].apply(extract_person_entities)
df.info()
# 5. Save output
df.to_csv("datasets/processed/final_alldata_dataset_with_entities.csv", index=False)

print("Dataset processing completed! Saved as final_dataset_with_entities.csv")


# %%
df = pd.read_csv("./datasets/processed/final_alldata_dataset_with_entities.csv")

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
save_json(unique_entities, "./datasets/processed/all_data_unique_entities.json")
# %%
df = pd.read_csv("all_data_news_captions.csv")
test_images = [get_image_path(id) for id in df['image_filename']]
valid_indices = [i for i, img in enumerate(test_images) if img is not None]
test_images = [test_images[i] for i in valid_indices]
df_test = df.iloc[valid_indices].reset_index(drop=True)
df_test.to_csv("datasets/processed/all_data_test_news_with_captions.csv", index=False)
# %%
df1 = pd.read_csv("datasets/processed/all_data_train.csv")
df2 = pd.read_csv("datasets/processed/all_data_test_news_with_captions.csv")
df1.rename(columns={'type': 'label'}, inplace=True)
df2.rename(columns={'type': 'label'}, inplace=True)

# Map 'Real' to 1 and 'Fake' to 0
mapping = {'real': 1, 'fake': 0}
df1['label'] = df1['label'].map(mapping)
df2['label'] = df2['label'].map(mapping)

# Optional: check the changes
df1.to_csv("datasets/processed/all_data_train.csv", index=False)
df2.to_csv("datasets/processed/all_data_test_news_with_captions.csv", index=False)
# %%
import pandas as pd

df2 = pd.read_csv("datasets/processed/all_data_test_news_with_captions.csv")

columns_to_keep = [
    "clean_title",
    "label",
    "image_filename",
    "entities",
    "image_path",
    "detected_persons",
    "num_faces",
    "person_names",
    "caption",
    "status"
]

df2 = df2[columns_to_keep]
df2.to_csv("datasets/processed/all_data_test_news_with_captions.csv", index=False)
# %%
import pandas as pd
df1 = pd.read_csv("datasets/processed/all_data_train.csv")
df2 = pd.read_csv("datasets/processed/all_data_caption_analysis_summary.csv")
print(len(df1))
print(len(df2))
# %%
