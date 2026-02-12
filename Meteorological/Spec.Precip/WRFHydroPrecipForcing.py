# -*- encoding: utf-8 -*-
'''
@File    :   spec_precip.py
@Create  :   2025-04-09 19:27:34
@Author  :   shihx2003
@Version :   1.0
@Contact :   shihx2003@outlook.com
'''
"China_Hourly_Merged_Precipitation_Analysis(CHMPA)_Data"

import os
import zipfile
import numpy as np
import xesmf as xe
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
from datetime import datetime, timedelta

class WRFHydroPrecipForcing:
    def __init__(self, geo_em=None, geo_sm=None, **kwargs):

        self.geo_em = geo_em if geo_em else "./geo_em.d03.nc"
        self.geo_sm = geo_sm if geo_sm else "./GEOGRID_LDASOUT_Spatial_Metadata.nc"
        self.description = kwargs.get('description', '')
        self.geo_em_ds, self.geo_sm_ds= self._read_geo()

    def _read_geo(self):

        geo_em_ds = xr.open_dataset(self.geo_em)
        geo_em_ds = geo_em_ds.rename({'XLONG_M': 'lon', 'XLAT_M': 'lat'})
        geo_em_ds['lat'] = geo_em_ds['lat'].sel(Time=0, drop=True)
        geo_em_ds['lon'] = geo_em_ds['lon'].sel(Time=0, drop=True)
        geo_sm_ds = xr.open_dataset(self.geo_sm, autoclose=True)

        return geo_em_ds, geo_sm_ds
    
    def precip_regrid(self, forcing):

        regridder = xe.Regridder(forcing, self.geo_em_ds, 'bilinear')
        forcing_regrid = regridder(forcing.precip_rate)
        forcing_regrid.coords['west_east'] = self.geo_sm_ds.x.values
        forcing_regrid.coords['south_north'] = self.geo_sm_ds.y.values
        forcing_regrid = forcing_regrid.rename({'west_east': 'x', 'south_north': 'y'})
        forcing_regrid.attrs['esri_pe_string'] = self.geo_sm_ds.crs.attrs['esri_pe_string']
        forcing_regrid.attrs['units'] = 'mm/s'

        return forcing_regrid

    def save_PRECIP_FORCING(self, forcing_regrid, save_path):
        dates = pd.to_datetime(forcing_regrid.time.values)
        for i in range(dates.size):
            str = dates[i].strftime('%Y%m%d%H')
            precip_forcing = forcing_regrid.isel(time=[i]).to_dataset(name="precip_rate")
            precip_forcing.attrs['units'] = "mm/s"
            precip_forcing.attrs['description'] = self.description
            precip_forcing.attrs["history"] = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            precip_forcing.to_netcdf(os.path.join(save_path, f'{str}00.PRECIP_FORCING.nc'))

    def regrid(self, forcing, save_path=None):
        if save_path==None:
            save_path = os.path.join(os.getcwd(), "output")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        forcing_regrid = self.precip_regrid(forcing)
        self.save_PRECIP_FORCING(forcing_regrid, save_path)
        return forcing_regrid

if __name__ == "__main__":
    # Example usage
    start_time = datetime(2023, 7, 1, 0)
    end_time = datetime(2023, 7, 5, 23)

    nc_dir = "F:/ART_1km"
    out_path = "./output_files"

    precip_ds = xr.open_mfdataset(os.path.join(nc_dir, "*.nc"), combine='by_coords')

    regrid = WRFHydroPrecipForcing(geo_em="./geo_em.d03.nc", out_path=out_path, description="1km-grid Analysis Real Time (ART_1km) precipitation")
    regrid.regrid(precip_ds)