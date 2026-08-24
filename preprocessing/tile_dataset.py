"""
Tile the orthomosaic and mask together to form the training dataset.
Using /data/masks as input and /data/tiles as output.
"""

tile = 512
stride = 256

# Iterate through image and mask pairs in the dataset
