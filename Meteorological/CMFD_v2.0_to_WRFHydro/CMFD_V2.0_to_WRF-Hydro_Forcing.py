# -*- encoding: utf-8 -*-
'''
@File    :   CMFD_V2.0_to_WRF-Hydro_Forcing.py
@Create  :   2026-01-26 13:44:00
@Author  :   shihx2003
@Version :   1.0
@Contact :   shihx2003@outlook.com
'''

# here put the import lib
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
import csv

def open_cmfd_month(cmfd_dir, mon):
    SRad_path = f"{cmfd_dir}/SRad/srad_CMFD_V0200_B-01_03hr_010deg_{mon:06d}.nc"
    LRad_path = f"{cmfd_dir}/LRad/lrad_CMFD_V0200_B-01_03hr_010deg_{mon:06d}.nc"
    SHum_path = f"{cmfd_dir}/SHum/shum_CMFD_V0200_B-01_03hr_010deg_{mon:06d}.nc"
    Temp_path = f"{cmfd_dir}/Temp/temp_CMFD_V0200_B-01_03hr_010deg_{mon:06d}.nc"
    Pres_path = f"{cmfd_dir}/Pres/pres_CMFD_V0200_B-01_03hr_010deg_{mon:06d}.nc"
    Wind_path = f"{cmfd_dir}/Wind/wind_CMFD_V0200_B-01_03hr_010deg_{mon:06d}.nc"
    Prec_path = f"{cmfd_dir}/Prec/prec_CMFD_V0200_B-01_03hr_010deg_{mon:06d}.nc"  

    ds_srad = xr.open_dataset(SRad_path)
    ds_lrad = xr.open_dataset(LRad_path)
    ds_shum = xr.open_dataset(SHum_path)
    ds_temp = xr.open_dataset(Temp_path)
    ds_pres = xr.open_dataset(Pres_path)
    ds_wind = xr.open_dataset(Wind_path)
    ds_prec = xr.open_dataset(Prec_path)

    ds = xr.merge(
        [
            ds_srad[["srad"]],
            ds_lrad[["lrad"]],
            ds_shum[["shum"]],
            ds_temp[["temp"]],
            ds_pres[["pres"]],
            ds_wind[["wind"]],
            ds_prec[["prec"]],
        ],
        compat="override",  # 属性冲突时用前者覆盖, 避免因为 global attrs 不同报错
        join="exact"        # 要求 time/lat/lon 完全一致；如果不一致会立刻提醒
    )

    return ds



def CMFD_interp_3h_to_1h(ds_3h: xr.Dataset,
                        vars_cont=("srad", "lrad", "shum", "temp", "pres", "wind"),
                        prec_var="prec",
                        method_cont="linear",
                        precip_method="repeat",
                        extend_last_window=True) -> xr.Dataset:
    """
    3h -> 1h。修复版：统一 time_1h，避免月底最后两小时缺失，同时避免 time 长度不一致报错。
    """
    if "time" not in ds_3h.coords:
        raise ValueError("ds_3h must have coordinate 'time'")

    # 确保 time 是 datetime64
    if not np.issubdtype(ds_3h["time"].dtype, np.datetime64):
        ds_3h = ds_3h.copy()
        ds_3h["time"] = xr.decode_cf(ds_3h).time
        if not np.issubdtype(ds_3h["time"].dtype, np.datetime64):
            raise ValueError("ds_3h['time'] must be datetime64 after decoding")

    t0 = pd.to_datetime(ds_3h.time.values[0])
    t1_raw = pd.to_datetime(ds_3h.time.values[-1])

    # 关键：只延长一次（+2小时），确保有22/23点
    t1 = t1_raw + pd.Timedelta(hours=2) if extend_last_window else t1_raw

    time_1h = pd.date_range(t0, t1, freq="1h")

    # 连续变量：线性插值 + 末端外推（补齐最后两小时）
    ds_cont_1h = ds_3h[list(vars_cont)].interp(
        time=time_1h,
        method=method_cont,
        kwargs={"fill_value": "extrapolate"}
    )

    # 降水
    if prec_var in ds_3h.data_vars:
        if precip_method == "linear":
            ds_prec_1h = ds_3h[[prec_var]].interp(
                time=time_1h,
                method="linear",
                kwargs={"fill_value": "extrapolate"}
            )
            return xr.merge([ds_cont_1h, ds_prec_1h], compat="override")

        if precip_method != "repeat":
            raise ValueError("precip_method must be 'repeat' or 'linear'")

        # repeat: 每个3h值重复3次，严格裁剪到 time_1h 长度
        prec3 = ds_3h[prec_var].values  # (time3h, lat, lon)
        prec1_vals = np.repeat(prec3, 3, axis=0)  # (time3h*3, lat, lon)

        # 裁剪到目标长度（关键）
        prec1_vals = prec1_vals[:len(time_1h), :, :]

        # 如果重复后的长度反而不够（理论上不该发生），则报错提示
        if prec1_vals.shape[0] != len(time_1h):
            raise ValueError(f"prec1_vals length {prec1_vals.shape[0]} != time_1h length {len(time_1h)}")

        prec_1h = xr.DataArray(
            prec1_vals,
            dims=("time", "lat", "lon"),
            coords={"time": time_1h, "lat": ds_3h["lat"], "lon": ds_3h["lon"]},
            name=prec_var,
            attrs=ds_3h[prec_var].attrs
        )

        return xr.merge([ds_cont_1h, prec_1h.to_dataset()], compat="override")

    return ds_cont_1h



def get_regridder(ds_src: xr.Dataset, 
                  ds_tgt: xr.Dataset, 
                  method="bilinear") -> xe.Regridder:
    grid_src = xr.Dataset({"lat": ds_src["lat"], "lon": ds_src["lon"]})
    grid_tgt = xr.Dataset(
        {
            "lat": (("y", "x"), ds_tgt["lat"].values),
            "lon": (("y", "x"), ds_tgt["lon"].values),
        }
    )
    regridder = xe.Regridder(grid_src, grid_tgt, method, reuse_weights=False)
    
    return regridder

def regrid_cmfd_to_ldasin(
    ds_src: xr.Dataset,          # 单时次：lat, lon 维（或 time 已被 isel 掉）
    regridder: xe.Regridder,
    var_names: list,             # 传 CMFD 侧变量名列表，例如 ["srad","lrad",...]
    fill_value: float = -9999.0,
    cmfd_fill_threshold: float = 1e19,
) -> xr.Dataset:
    """
    将单时次 CMFD 数据集重网格到 LDASIN 网格，并按 WRF-Hydro/LDASIN 命名输出。

    变量对应关系（CMFD -> LDASIN）：
      srad -> SWDOWN
      lrad -> LWDOWN
      shum -> Q2D
      temp -> T2D
      pres -> PSFC
      wind -> U2D, V2D   (U2D=wind*cos45, V2D=wind*cos45)
      prec -> RAINRATE   (kg m-2 s-1 等价于 mm s-1)

    注意：
    - ds_src 建议是单时次切片：ds.isel(time=i).drop_vars('time') 或 ds.isel(time=i)
      若仍含 time 维，输出也会保留 time（长度=1）。
    - CMFD 缺测值一般是 1e20，这里用阈值过滤成 NaN，再插值，最后填 fill_value。
    """

    mapping = {
        "SRad": "SWDOWN",
        "LRad": "LWDOWN",
        "SHum": "Q2D",
        "Temp": "T2D",
        "Pres": "PSFC",
        "Wind": ("U2D", "V2D"),
        "Prec": "RAINRATE",
        # 如果你实际变量名是小写（CMFD原生）：也给一套别名，方便直接用
        "srad": "SWDOWN",
        "lrad": "LWDOWN",
        "shum": "Q2D",
        "temp": "T2D",
        "pres": "PSFC",
        "wind": ("U2D", "V2D"),
        "prec": "RAINRATE",
    }

    out = xr.Dataset()
    c45 = np.cos(np.deg2rad(45.0))

    for v in var_names:
        if v not in ds_src:
            raise KeyError(f"Variable '{v}' not found in ds_src. Available: {list(ds_src.data_vars)}")

        tgt = mapping.get(v, None)
        if tgt is None:
            raise KeyError(f"No mapping rule defined for CMFD variable '{v}'")

        da = ds_src[v]

        # 处理 CMFD 缺测：1e20 -> NaN
        # 对 float 数据这样做最稳
        if np.issubdtype(da.dtype, np.floating):
            da = da.where(da < cmfd_fill_threshold)

        # Wind 特殊：生成 U2D/V2D
        if isinstance(tgt, tuple):

            wind_rg = regridder(da)
            # U2D/V2D = wind * cos(45°)
            u = wind_rg * c45
            vcomp = wind_rg * c45

            u = u.where(np.isfinite(u), fill_value)
            vcomp = vcomp.where(np.isfinite(vcomp), fill_value)

            u.name = "U2D"
            vcomp.name = "V2D"

            u.attrs.update({
                "long_name": "Near surface wind in the u-component (scaled from wind speed)",
                "units": "m s-1",
                "missing_value": fill_value,
                "remap": "bilinear",
            })
            vcomp.attrs.update({
                "long_name": "Near surface wind in the v-component (scaled from wind speed)",
                "units": "m s-1",
                "missing_value": fill_value,
                "remap": "bilinear",
            })

            out["U2D"] = u
            out["V2D"] = vcomp
            continue

        # 其他变量：直接重网格并重命名
        da_rg = regridder(da)
        da_rg = da_rg.where(np.isfinite(da_rg), fill_value)

        da_rg.name = tgt

        da_rg.attrs.update({
            "missing_value": fill_value,
            "remap": "bilinear",
        })

        if tgt == "SWDOWN":
            da_rg.attrs.update({"units": "W m-2", "long_name": "Downward short-wave radiation flux"})
        elif tgt == "LWDOWN":
            da_rg.attrs.update({"units": "W m-2", "long_name": "Downward long-wave radiation flux"})
        elif tgt == "Q2D":
            da_rg.attrs.update({"units": "kg kg-1", "long_name": "Specific humidity"})
        elif tgt == "T2D":
            da_rg.attrs.update({"units": "K", "long_name": "Temperature"})
        elif tgt == "PSFC":
            da_rg.attrs.update({"units": "Pa", "long_name": "Pressure"})
        elif tgt == "RAINRATE":
            da_rg.attrs.update({"units": "mm s^-1", "long_name": "RAINRATE"})

        out[tgt] = da_rg
    out.attrs["forcing_source"] = "China Meteorological Forcing Dataset (CMFD) v2.0"
    return out

def wrap_to_ldasin(
    out_rg: xr.Dataset,          # 你重网格后的结果：变量在 (y,x) 或 (time,y,x)
    ds_tgt: xr.Dataset,          # 目标LDASIN网格文件打开后的 ds（提供lat/lon与网格大小）
    tstamp,                      # datetime-like, e.g. "2023-05-01 00:00:00"
    fill_value: float = -9999.0,
    v2d_fill_value: float = 9.969209968386869e36,  # 你示例里V2D的FillValue
):
    """
    把重网格结果包装成 WRF-Hydro LDASIN_DOMAIN3 风格的 Dataset（Time=1）。

    out_rg 需要包含变量：
      SWDOWN, LWDOWN, Q2D, T2D, PSFC, U2D, V2D, RAINRATE
    且维度为 (y,x) 或 (time,y,x)（time长度=1）。
    """

    # ---- time 处理 ----
    t = pd.to_datetime(tstamp)
    timestr = t.strftime("%Y-%m-%d_%H:%M:%S")
    valid_sec = (t - pd.Timestamp("1970-01-01")).total_seconds()

    # ---- 取二维场 (y,x) ----
    def _to_yx(da):
        if "time" in da.dims:
            da = da.isel(time=0)
        if "Time" in da.dims:
            da = da.isel(Time=0)
        return da

    # ---- 输出维度大小来自目标网格 ----
    ny, nx = ds_tgt["lat"].shape

    # ---- 构造输出 Dataset ----
    ds = xr.Dataset()

    ds = ds.assign_coords(
        Time=("Time", [0]),
        south_north=("south_north", np.arange(ny, dtype=np.int32)),
        west_east=("west_east", np.arange(nx, dtype=np.int32)),
    )

    # lat/lon
    ds["lat"] = xr.DataArray(
        ds_tgt["lat"].values.astype("float32"),
        dims=("south_north", "west_east"),
        attrs=ds_tgt["lat"].attrs
    )
    ds["lon"] = xr.DataArray(
        ds_tgt["lon"].values.astype("float32"),
        dims=("south_north", "west_east"),
        attrs=ds_tgt["lon"].attrs
    )

    # Times: char(DateStrLen=20) —— 这里做成一个长度20的字符数组
    ds = ds.assign_coords(DateStrLen=np.arange(20, dtype=np.int32))
    ds["Times"] = xr.DataArray(
        np.array(list(timestr.ljust(20)[:20]), dtype="S1"),
        dims=("DateStrLen",)
    )

    # valid_time(Time)
    ds["valid_time"] = xr.DataArray(
        np.array([valid_sec], dtype="float64"),
        dims=("Time",),
        attrs={
            "calendar": "standard",
            "units": "seconds since 1970-01-01 00:00:00"
        }
    )

    # ---- 强迫变量写入 (Time, south_north, west_east) ----
    # 需要的变量列表（顺序无所谓）
    need = ["T2D","Q2D","U2D","V2D","PSFC","RAINRATE","SWDOWN","LWDOWN"]
    for v in need:
        if v not in out_rg:
            raise KeyError(f"out_rg missing variable: {v}")

        da = _to_yx(out_rg[v])

        # 确保是 (y,x)
        if da.dims != ("y","x"):
            da = da.transpose("y","x")

        arr = da.values
        # 缺测处理：NaN -> fill_value / v2d_fill_value
        if v == "V2D":
            arr = np.where(np.isfinite(arr), arr, v2d_fill_value)
        else:
            arr = np.where(np.isfinite(arr), arr, fill_value)

        ds[v] = xr.DataArray(
            arr[np.newaxis, :, :].astype("float64" if v != "RAINRATE" else "float32"),
            dims=("Time", "south_north", "west_east"),
        )

    # ---- 设置属性（尽量贴近你示例）----
    ds["T2D"].attrs.update({
        "standard_name": "air_temperature",
        "long_name": "Temperature",
        "units": "K",
        "missing_value": np.float32(fill_value),
        "cell_methods": "time: point",
        "remap": "remapped via ESMF_regrid_with_weights: Bilinear",
    })
    ds["Q2D"].attrs.update({
        "standard_name": "specific_humidity",
        "long_name": "Specific humidity",
        "units": "kg kg-1",
        "missing_value": np.float32(fill_value),
        "cell_methods": "time: point",
        "remap": "remapped via ESMF_regrid_with_weights: Bilinear",
    })
    ds["U2D"].attrs.update({
        "standard_name": "wind_speed",
        "long_name": "Wind speed",
        "units": "m s-1",
        "missing_value": np.float32(fill_value),
        "cell_methods": "time: point",
        "remap": "remapped via ESMF_regrid_with_weights: Bilinear",
    })
    ds["V2D"].attrs.update({
        "missing_value": np.float32(v2d_fill_value),
        "remap": "remapped via ESMF_regrid_with_weights: Bilinear",
    })
    ds["PSFC"].attrs.update({
        "standard_name": "surface_air_pressure",
        "long_name": "Pressure",
        "units": "Pa",
        "missing_value": np.float32(fill_value),
        "cell_methods": "time: point",
        "remap": "remapped via ESMF_regrid_with_weights: Bilinear",
    })
    ds["RAINRATE"].attrs.update({
        "standard_name": "precipitation_flux",
        "long_name": "RAINRATE",
        "description": "RAINRATE",
        "units": "mm s^-1",
        "missing_value": np.float32(fill_value),
        "cell_methods": "time: mean",
        "remap": "remapped via ESMF_regrid_with_weights: Bilinear",
    })
    ds["SWDOWN"].attrs.update({
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "long_name": "Downward short-wave radiation flux",
        "units": "W m-2",
        "missing_value": np.float32(fill_value),
        "cell_methods": "time: mean",
        "remap": "remapped via ESMF_regrid_with_weights: Bilinear",
    })
    ds["LWDOWN"].attrs.update({
        "standard_name": "surface_downwelling_longwave_flux_in_air",
        "long_name": "Downward long-wave radiation flux",
        "units": "W m-2",
        "missing_value": np.float32(fill_value),
        "cell_methods": "time: mean",
        "remap": "remapped via ESMF_regrid_with_weights: Bilinear",
    })

    return ds

def varmaxmin(ds: xr.Dataset,
              file_name: str = "",
              fill_value: float = -9999.0,
              v2d_fill_value: float = 9.969209968386869e36) -> pd.DataFrame:
    ldasin_var_names = ["SWDOWN", "LWDOWN", "Q2D", "T2D", "PSFC", "U2D", "V2D", "RAINRATE"]
    records = []

    # 时间戳（从 valid_time 取）
    tstamp = ""
    if "valid_time" in ds:
        try:
            t = pd.to_datetime(ds["valid_time"].values[0], unit="s", origin="unix")
            tstamp = t.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            tstamp = ""

    for v in ldasin_var_names:
        if v not in ds:
            continue

        da = ds[v]
        if "Time" in da.dims:
            da = da.isel(Time=0)

        a = da.values
        fv = v2d_fill_value if v == "V2D" else fill_value

        n_total = a.size
        n_nan = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
        n_fill = int((a == fv).sum()) if np.issubdtype(a.dtype, np.floating) else 0

        mask = np.isfinite(a) & (a != fv) if np.issubdtype(a.dtype, np.floating) else np.ones_like(a, dtype=bool)
        if mask.any():
            amin = float(a[mask].min())
            amax = float(a[mask].max())
            amean = float(a[mask].mean())
        else:
            amin = np.nan
            amax = np.nan
            amean = np.nan

        # 简单范围 QC
        qc_flag = 0
        if mask.any():
            if v == "SWDOWN" and (amin < -1e-6 or amax > 1500): qc_flag = 1
            if v == "LWDOWN" and (amin < -1e-6 or amax > 800):  qc_flag = 1
            if v == "T2D"    and (amin < 180 or amax > 340):    qc_flag = 1
            if v == "Q2D"    and (amin < 0 or amax > 0.06):     qc_flag = 1
            if v == "PSFC"   and (amin < 30000 or amax > 110000): qc_flag = 1
            if v == "RAINRATE" and (amin < -1e-12 or amax > 0.02): qc_flag = 1
            if v in ("U2D", "V2D") and (amin < -60 or amax > 60): qc_flag = 1

        records.append({
            "file": file_name,
            "time": tstamp,
            "var": v,
            "min": amin,
            "max": amax,
            "mean": amean,
            "nan_count": n_nan,
            "fill_count": n_fill,
            "total_count": n_total,
            "qc_flag": qc_flag
        })

    return pd.DataFrame.from_records(records)

def get_lonlat_range(ldasin_path):
    ldasin_ds = xr.open_dataset(ldasin_path)
    lon_min = ldasin_ds['lon'].min().item() - 1.0
    lon_max = ldasin_ds['lon'].max().item() + 1.0
    lat_min = ldasin_ds['lat'].min().item() - 1.0
    lat_max = ldasin_ds['lat'].max().item() + 1.0

    print(f"Longitude range: {lon_min} to {lon_max}")
    print(f"Latitude range: {lat_min} to {lat_max}")

    return lon_min, lon_max, lat_min, lat_max


def main():
    diag_list = []
    for mon in [202305,202306,202307,202308,202309]:
        cmfd_var_names = ["SRad", "LRad", "SHum", "Temp", "Pres", "Wind", "Prec"]
        ldasin_var_names = ["SWDOWN", "LWDOWN", "Q2D", "T2D", "PSFC", "U2D", "V2D", "RAINRATE"]
        cmfd_dir   = "./CMFD_V2.0_Data_forcing_03hr_010deg"         # SRad/srad_CMFD_V0200_B-01_03hr_010deg_202305.nc
        ldasin_path = "./Zijinguan.LDASIN_DOMAIN3"

        outdir = "./To_WRF-Hydro_Forcing_CMFD_1h"

        csv_path = f"./diagnostics.csv"

        lon_min, lon_min, lon_max, lat_min, lat_max = get_lonlat_range(ldasin_path)

        cmfd_ds = open_cmfd_month(cmfd_dir, mon)
        cmfd_ds = cmfd_ds.sel(
                            lon=slice(lon_min, lon_max),
                            lat=slice(lat_min, lat_max)
                                    )
        cmfd_ds_1h = CMFD_interp_3h_to_1h(
                            cmfd_ds,
                            vars_cont=["srad", "lrad", "shum", "temp", "pres", "wind"],
                            prec_var="prec",
                            method_cont="linear",
                            precip_method="repeat",
                            extend_last_window=True
                                    )

        ldasin_ds = xr.open_dataset(ldasin_path, engine="netcdf4")
        regridder = get_regridder(cmfd_ds_1h, ldasin_ds, method="bilinear")
        
        for i in range(cmfd_ds_1h.sizes["time"]):
            ds_1t = cmfd_ds_1h.isel(time=i)
            cmfd_var_names = ["srad","lrad","shum","temp","pres","wind","prec"]
            out_rg = regrid_cmfd_to_ldasin(ds_1t, regridder, cmfd_var_names)

            tstamp = cmfd_ds_1h["time"].values[i]
            out_ldasin = wrap_to_ldasin(out_rg, ldasin_ds, tstamp)

            ts = pd.to_datetime(tstamp).strftime("%Y%m%d%H")
            out_path = f"{outdir}/{ts}.LDASIN_DOMAIN3"
            out_ldasin.to_netcdf(out_path, format="NETCDF3_CLASSIC", engine="scipy")
            # 诊断：先存起来，不写盘
            df_diag = varmaxmin(out_ldasin, file_name=os.path.basename(out_path))
            diag_list.append(df_diag)

            if (df_diag["qc_flag"] == 1).any():
                print("QC WARNING:", out_path)

            print("Wrote:", out_path)

        # 循环结束后：一次性写 CSV
    diag_all = pd.concat(diag_list, ignore_index=True)
    csv_path = f"{outdir}/diagnostics_{mon:06d}.csv"
    diag_all.to_csv(csv_path, index=False)
    print("Diagnostics saved:", csv_path)
    
if __name__ == "__main__":
    main()