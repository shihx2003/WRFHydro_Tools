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

class PrecipDataLoader:
    def __init__(self, zips_dir, grid_info, first_name=None, last_name=None):

        self.zips_dir = zips_dir
        self.first_name = first_name if first_name else ''
        self.last_name = last_name if last_name else '.' + grid_info['file_format']
        self.grid_info = grid_info
        self.file_format = self.grid_info['file_format']
        self.description = grid_info.get('description', '')
        

        self.lon, self.lat = self._lon_lat()

    def _lon_lat(self):

        lon_min = self.grid_info['lon_min']
        lon_max = self.grid_info['lon_max']
        lat_min = self.grid_info['lat_min']
        lat_max = self.grid_info['lat_max']
        grid_res = self.grid_info['grid_res']

        lon = (np.arange(lon_min*100, (lon_max + grid_res)*100, grid_res*100)) / 100
        lat = (np.arange(lat_min*100, (lat_max + grid_res)*100, grid_res*100)) / 100
        print(len(lon), len(lat))
        return lon, lat
    
    def precip_info(self, start_time, end_time):
        file_names = []
        times = []
        while start_time <= end_time:
            time_str = start_time.strftime("%Y%m%d%H")
            name =self.first_name + time_str + self.last_name
            file_names.append(name)
            times.append(start_time)
            start_time += timedelta(hours=1)
        files_info = np.array(list(zip(file_names, times)))
        return files_info
    
    def readzip(self, zip_path, name):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"File {zip_path} does not exist.")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(name) as file:
                content = file.read()
                return content
    
    def read_grid(self, zip_path, path):
        
        if self.file_format == "m4":
            return self._read_m4(zip_path, path)
        elif self.file_format == "grd":
            return self._read_grd(zip_path, path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def _read_m4(self, zip_path, m4_path):
        from io import BytesIO
        content = self.readzip(zip_path, m4_path)
        file = BytesIO(content)
        header = file.readline().strip()
        file.readline()
        grid_data = []
        for line in file:
            grid_data.extend(map(float, line.strip().split()))
        grid_array = np.array(grid_data).reshape(len(self.lat), len(self.lon))
        grid_array = np.nan_to_num(grid_array, nan=0.0)
        
        return grid_array
    
    def _read_grd(self, zip_path, grd_path):
        content = self.readzip(zip_path, grd_path)
        grid_data = np.frombuffer(content, dtype=np.float32)
        grid_data = grid_data.reshape(2, len(self.lat), len(self.lon))
        grid_data=np.maximum(grid_data[0, :, :], 0)

        return grid_data

    def process_grid_data(self, file_info, year):

        zip_path = os.path.join(self.zips_dir, year + ".zip")
        grd_path = f'{year}/{file_info[0]}'
        time = file_info[1]

        grid_value = self.read_grid(zip_path, grd_path)
        grid_ds = xr.Dataset(
            {
                "precip": (["time", "lat", "lon"], grid_value[np.newaxis, :, :]),
                "precip_rate": (["time", "lat", "lon"], grid_value[np.newaxis, :, :] / (60.0*60.0))
            },
            coords={
                "time": [time], # Convert to UTC ?
                "lon": self.lon,
                "lat": self.lat
            }
        )
        grid_ds["precip"].attrs["units"] = "mm/h"
        grid_ds["precip"].attrs["description"] = self.description
        grid_ds["precip_rate"].attrs["units"] = "mm/s"
        grid_ds["precip_rate"].attrs["description"] = self.description
        grid_ds.attrs["time_zone"] = "UTC"
        grid_ds.attrs["resolution"] = f"{self.grid_info['grid_res']}° x {self.grid_info['grid_res']}°"
        grid_ds.attrs["history"] = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return grid_ds

    def load(self, start, end):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        precip_file_info = self.precip_info(start, end)
        
        year = str(start.year)

        datasets = []
        results = Parallel(n_jobs=-1)(delayed(self.process_grid_data)(precip_file, year) for precip_file in precip_file_info)

        datasets.extend(results)
        precip_ds = xr.concat(datasets, dim="time")

        # precip_ds['precip_rate'] = precip_ds.precip / (60.0*60.0)
        precip_rate_ds = precip_ds.drop_vars('precip')

        return precip_rate_ds


if __name__ == "__main__":
    # Example usage
    start_time = datetime(2023, 7, 1, 0)
    end_time = datetime(2023, 7, 5, 23)

    zips_dir = "F:/ART_1km"
    out_path = "./output_files"

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
    precip_ds = precip_loader.load(start_time, end_time)
    print(precip_ds)

    regrid = WRFHydroPrecipForcing(geo_em="./geo_em.d03.nc", out_path=out_path, description="1km-grid Analysis Real Time (ART_1km) precipitation")
    regrid.regrid(precip_ds)