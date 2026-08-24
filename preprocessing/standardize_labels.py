"""
For the columns in the shape files that are equivalent ("Cover", "Field", "Category"), standardize the labels
I.e. The labels should be submergent, emergent, and loamy soil but some are misspelled or have different variations. For example, "submergent" could be spelled as "submergent", "submergnt", "submergnt soil", etc. 
The goal is to standardize these labels to the correct spelling and format.
"""

import os
import geopandas as gpd
import pandas as pd


def get_label_column(gdf):
    for col in gdf.columns:
        if col.lower() in ['cover', 'field', 'category']:
            return col
    
    raise ValueError(f"No label column found. Columns: {gdf.columns.tolist()}")

def standardize_labels(gdf):
    label_col = get_label_column(gdf)
    
    def standardize_label(label):
        if pd.isnull(label):
            return label
        label_lower = label.lower()
        if 'sub' in label_lower:
            return 'submergent'
        elif 'em' in label_lower:
            return 'emergent'
        elif 'loam' in label_lower:
            return 'loamy soil'
        else:
            return label  # Return the original label if no match is found
    
    gdf[label_col] = gdf[label_col].apply(standardize_label)
    return gdf


