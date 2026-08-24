"""
Rasterize polygons from shapefiles to create binary masks for each class.
Using the data folder structure:
data/
  ├── village1_folder_w_shp_file/
  │   ├── village1_shapefile.shp
  │   ├── village1_shapefile.dbf
  │   ├── village1_shapefile.prj
  │   └── village1_shapefile.shx
  ├── village2_folder_w_shp_file/
  │   ├── village2_shapefile.shp
and so on for each village folder.
"""

import os
import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np

def rasterize_shapefile(shapefile, output_path, resolution=1):
    # Read the shapefile
    gdf = gpd.read_file(shapefile)
    bounds = gdf.total_bounds  
    min_x, min_y, max_x, max_y = bounds
    
    # Define raster dimensions
    width = int((max_x - min_x) / resolution)
    height = int((max_y - min_y) / resolution)
    
    mask_array = np.zeros((height, width), dtype=np.uint8)
    
    # Rasterize polygons
    for _, row in gdf.iterrows():
        geom = row['geometry']
        mask_array = np.maximum(
            mask_array,
            features.rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height),
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
        )
    
    # Write the raster to a GeoTIFF file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=gdf.crs,
        transform=rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, width, height),
    ) as dst:
        dst.write(mask_array, 1)

def main(input_dir='data', output_dir='data/masks', resolution=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through each village folder
    for village_folder in os.listdir(input_dir):
        village_path = os.path.join(input_dir, village_folder)
        
        if os.path.isdir(village_path):
            shapefile = next((os.path.join(village_path, f) for f in os.listdir(village_path) if f.endswith('.shp')), None)
            
            if shapefile:
                print(f"Processing {village_folder}...")
                mask_output_path = os.path.join(output_dir, f"{village_folder}_mask.tif")
                rasterize_shapefile(shapefile, mask_output_path, resolution)
                print(f"Rasterized mask saved to {mask_output_path}")
            else:
                print(f"No shapefile found in {village_folder}")

if __name__ == "__main__":
    main()



