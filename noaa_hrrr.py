from datetime import date, timedelta
import tempfile
import xarray as xr
import requests
import timeit
import os
import warnings
import config
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cmocean

warnings.simplefilter("ignore")


def noaa_hrrr(day, parameter_name, AOI_lat, AOI_lon, LN):
    # Constants for creating the full URL
    blob_container = "https://storage.googleapis.com/high-resolution-rapid-refresh"
    sector = "conus"
    _date = day
    cycle = 12  # noon
    forecast_hour = 0  # offset from cycle time
    product = "wrfsfcf"  # 2D surface levels

    # Put it all together
    file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
    url = f"{blob_container}/hrrr.{_date:%Y%m%d}/{sector}/{file_path}"

    # Fetch the idx file by appending the .idx file extension to our already formatted URL
    r = requests.get(f"{url}.idx")
    idx = r.text.splitlines()

    # Pluck the byte offset from this line, plus the beginning offset of the next line
    parameter_id = int([l for l in idx if parameter_name in l][0].split(":")[0])
    line_num = parameter_id - 1
    range_start = idx[line_num].split(":")[1]
    range_end = idx[parameter_id].split(":")[1]

    # read the file requested from url
    file = tempfile.NamedTemporaryFile(prefix="tmp_", delete=False)
    headers = {"Range": f"bytes={range_start}-{range_end}"}
    resp = requests.get(url, headers=headers, stream=True)
    with file as f:
        f.write(resp.content)
    ds = xr.open_dataset(file.name, engine='cfgrib', backend_kwargs={'indexpath': ''})
    ds_df = ds.to_dataframe().reset_index(drop=True)

    # subset the dataframe
    ds_df = ds_df[
        (AOI_lat[0] < ds_df['latitude']) & (ds_df['latitude'] < AOI_lat[1]) &
        (AOI_lon[0] + 360 < ds_df['longitude']) & (ds_df['longitude'] < AOI_lon[1] + 360)
        ]

    # obtain the feature for each LN segment
    name = ds_df.columns[len(ds_df.columns) - 1]
    ds_df = gpd.GeoDataFrame(
        getattr(ds_df, name), crs='epsg:4326',
        geometry=gpd.points_from_xy(ds_df.longitude - 360, ds_df.latitude)
    ).reset_index(drop=True)
    ds_Cartesian = ds_df.to_crs('EPSG:5234')
    LN_Cartesian = LN.to_crs('EPSG:5234')
    LN_ft = []
    for i in range(len(LN_Cartesian)):
        dist = ds_Cartesian.geometry.distance(LN_Cartesian.geometry[i])
        LN_ft.append(ds_df[name][np.argmin(dist)])

    return np.array(LN_ft)


# obtain the interested features from noaa hrrr model
def hrrr_data():
    hrrr = np.zeros((len(LN), days, 3))
    for i in range(days):
        day = start + timedelta(days=i)
        hrrr[:, i, 0] = noaa_hrrr(day, parameter_name1, AOI_lat, AOI_lon, LN) - 273.15
        hrrr[:, i, 1] = noaa_hrrr(day, parameter_name2, AOI_lat, AOI_lon, LN)
        hrrr[:, i, 2] = noaa_hrrr(day, parameter_name3, AOI_lat, AOI_lon, LN)
    cwd = os.getcwd()
    hrrr_path = os.path.join(cwd, 'Data', 'hrrr3b.npy')
    np.save(hrrr_path, hrrr)  # save data


def hrrr_plot():
    LN_ft = noaa_hrrr(day, parameter_name, AOI_lat, AOI_lon, LN) - 273.15
    fig, ax = plt.subplots(1, figsize=(10, 10))
    LN.plot(ax=ax, column=np.array(LN_ft), legend=True, linewidth=3, cmap='plasma')
    plt.xlabel('long', fontsize=26, color='black')
    plt.ylabel('lat', fontsize=26, color='black')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=18)
    plt.savefig('tmp.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()


if __name__ == "__main__":
    # show an illustrative plot of hrrr data on LN
    start_time = timeit.default_timer()
    day = date(2019, 6, 1)
    parameter_name = 'TMP:2 m'
    AOI_lat = [36, 37]
    AOI_lon = [-120, -119]
    LN = config.LN
    hrrr_plot()
    stop_time = timeit.default_timer()
    print('Time: ', stop_time - start_time)
    start_time = timeit.default_timer()
    parameter_name1 = 'TMP:2 m'
    parameter_name2 = 'WIND:10 m'
    parameter_name3 = 'SPFH:2 m'  # set interested parameters
    start = date(2019, 5, 29)
    end = date(2019, 7, 7)
    days = (end - start).days + 1  # set interested dates
    LN = config.LN  # input global variable
    hrrr_data()

    stop_time = timeit.default_timer()
    print('Time: ', stop_time - start_time)
