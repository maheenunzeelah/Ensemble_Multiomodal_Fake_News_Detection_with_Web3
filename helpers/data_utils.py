# helpers/data_utils.py
import pandas as pd
import os
from PIL import Image

def load_and_combine(train_path, val_path, test_path):
    """Load and combine train, validation, and test datasets."""
    train = pd.read_csv(train_path, sep="\t")
    val = pd.read_csv(val_path,sep="\t")
    test = pd.read_csv(test_path, sep="\t")
    df = pd.concat([train, val, test], ignore_index=True)

    # rename label column
    df = df.rename(columns={"2_way_label": "label"})
    return df


def keep_selected_columns(df):
    """Keep only clean_title, id, label."""
    return df[["clean_title", "id", "label"]]


def subsample_rows(df, n=12000):
    """Random subsample n rows."""
    return df.sample(n, random_state=42)


def subsample_balanced(df, fake_label=0, true_label=1, n_each=6000):
    """Balanced subsample n_each rows per class."""
    fake_df = df[df["label"] == fake_label]
    true_df = df[df["label"] == true_label]

    fake_sample = fake_df.sample(n_each, random_state=42)
    true_sample = true_df.sample(n_each, random_state=42)

    return pd.concat([fake_sample, true_sample]).sample(frac=1, random_state=42)



def get_image_path(image_id, base_path='allData_images'):
    """Find the actual image file with any extension. Returns None if corrupted."""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for ext in extensions:
        path = f"{base_path}/{image_id}{ext}"
        print(path)
        if os.path.exists(path):
            try:
                # Try to open and identify the image
                with Image.open(path) as img:
                    img.load()
                return path
            except Exception as e:
                print(f"Warning: Cannot identify image for ID {image_id}: {e}")
                return None
    
    print(f"Warning: Image file not found for ID {image_id}")
    return None