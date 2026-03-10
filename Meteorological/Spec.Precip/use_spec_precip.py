# -*- encoding: utf-8 -*-
'''
@File    :   NMC_m4_GridRain2WRFHydro_regrid_Spec.Precip.py
@Create  :   2025-04-10 22:26:45
@Author  :   shihx2003
@Version :   1.0
@Contact :   shihx2003@outlook.com

@Useage:
    python NMC_m4_GridRain2WRFHydro_regrid_Spec.Precip.py 2023
'''

# here put the import lib
import os
from datetime import datetime, timedelta
from spec_precip import PrecipDataLoader, WRFHydroPrecipForcing
import sys

def get_year():
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
        except ValueError:
            print("year must be an integer")
            sys.exit(1)
    else:
        print("Please provide a year argument")
        sys.exit(1)
    
    return year

zips_dir = "/public/home/Shihuaixuan/Data/NMC_01H_GridRain_Haihe"
out_path = "./Spec.Precip"

grid_info = {
    'lon_min': 111.7,
    'lon_max': 120.0,
    'lat_min': 35.0,
    'lat_max': 43.0,
    'grid_res': 0.01,
    'file_format': 'm4',
    'description': "1km-grid Analysis Real Time (ART_1km) precipitation"
}

precip_loader = PrecipDataLoader(zips_dir, grid_info, first_name='', last_name='.m4')
regrid = WRFHydroPrecipForcing(geo_em="./ZJG_geo_em.d03.nc", 
                               geo_sm="./ZJG_GEOGRID_LDASOUT_Spatial_Metadata.nc",
                               description="1km-grid Analysis Real Time (ART_1km) precipitation")

year = get_year()
jobs = {'1': {"start": datetime(year, 1, 1, 0), "end": datetime(year, 3, 31, 23)},
        '2': {"start": datetime(year, 4, 1, 0), "end": datetime(year, 6, 30, 23)},
        '3': {"start": datetime(year, 7, 1, 0), "end": datetime(year, 9, 30, 23)},
        '4': {"start": datetime(year, 10, 1, 0), "end": datetime(year, 12, 31, 23)},
}

for precip_name, time_range in jobs.items():
    start_time = time_range["start"]
    end_time = time_range["end"]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    precip_ds = precip_loader.load(start_time, end_time)
    regrid.regrid(precip_ds, save_path=out_path)