
import geopandas as gpd
from pathlib import Path

SHAPEFILES = [
    "00_Ancillary_Data/AbuDhabi_MixedUrban_Regions.shp",
    "00_Ancillary_Data/AbuDhabi_Regions.shp"
]

def inspect():
    for shp_path in SHAPEFILES:
        p = Path(shp_path)
        if not p.exists():
            print(f"Not found: {shp_path}")
            continue
            
        print(f"\n===== Inspecting: {p.name} =====")
        gdf = gpd.read_file(shp_path)
        print(f"CRS: {gdf.crs}")
        print(f"Columns: {gdf.columns.tolist()}")
        print(f"First 3 rows:\n{gdf.head(3)}")
        
        # Look for category columns
        potential_cats = ['LEGEND', 'CLASS', 'TYPE', 'NAME', 'DESCRIPTIO', 'LANDUSE']
        for col in potential_cats:
            if col in gdf.columns:
                print(f"\nUnique values in '{col}':")
                print(gdf[col].unique())

if __name__ == "__main__":
    inspect()
