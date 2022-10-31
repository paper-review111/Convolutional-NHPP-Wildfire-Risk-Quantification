import os
from datetime import date, timedelta
import rioxarray as rxr
import geopandas as gpd
import numpy as np
import warnings
import timeit
import pyproj
import config
import cmocean
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def modis_ndvi(day, AOI_lat, AOI_lon, search_size, LN):
    # read hdf file
    cwd = os.getcwd()
    file_name = str(day.timetuple().tm_yday) + '.hdf'
    file_path = os.path.join(cwd, 'Data', 'NDVI', file_name)
    desired_bands = ['sur_refl_b01_1', 'sur_refl_b02_1']
    modis_pre_bands = rxr.open_rasterio(file_path, masked=True, variable=desired_bands).squeeze()
    modis_crs = modis_pre_bands.rio.crs
    # print(modis_crs)

    # transform coordinates in epsg:4326
    lat = modis_pre_bands.to_dataframe().index.get_level_values(0)
    lon = modis_pre_bands.to_dataframe().index.get_level_values(1)
    modis_wgs = pyproj.Transformer.from_crs(modis_crs, 'EPSG: 4326')
    lat, lon = modis_wgs.transform(lon, lat)
    modis_ds = modis_pre_bands.to_dataframe().reset_index(drop=True).filter(items=['sur_refl_b01_1', 'sur_refl_b02_1'])
    modis_ds['lon'] = lon
    modis_ds['lat'] = lat
    # print(modis_ds)

    # filter the dataset in the first stage
    modis_ds = modis_ds[
        (AOI_lon[0] <= modis_ds['lon']) & (modis_ds['lon'] <= AOI_lon[1]) &
        (AOI_lat[0] <= modis_ds['lat']) & (modis_ds['lat'] <= AOI_lat[1])
        ].reset_index(drop=True)
    # print(modis_ds)

    # calculate NDVI
    NDVI_dn = np.array(modis_ds.sur_refl_b01_1) + np.array(modis_ds.sur_refl_b02_1)
    NDVI_n = np.array(modis_ds.sur_refl_b02_1) - np.array(modis_ds.sur_refl_b01_1)
    NDVI = NDVI_n / NDVI_dn
    NDVI[(NDVI > 1)] = np.nan
    NDVI[(NDVI < -1)] = np.nan
    modis_ds['NDVI'] = NDVI
    modis_ds['NDVI'] = modis_ds.NDVI.interpolate()
    # print(modis_ds)

    # extract NDVI for LN
    LN_NDVI = []
    for i in range(len(LN)):
        bounds = LN.geometry[i].bounds
        # create a smaller search area
        modis_ds_sub = modis_ds[
            (bounds[0] - search_size <= modis_ds['lon']) & (modis_ds['lon'] <= bounds[2] + search_size) &
            (bounds[1] - search_size <= modis_ds['lat']) & (modis_ds['lat'] <= bounds[3] + search_size)
            ].reset_index(drop=True)
        modis_ds_sub = gpd.GeoDataFrame(
            modis_ds_sub.NDVI, crs='epsg:4326',
            geometry=gpd.points_from_xy(modis_ds_sub.lon, modis_ds_sub.lat)
        )
        dist = modis_ds_sub.to_crs('EPSG:5234').geometry.distance(LN.to_crs('EPSG:5234').geometry[i])
        LN_NDVI.append(modis_ds_sub.NDVI[np.argmin(dist)])

    return LN_NDVI


def ndvi_data():
    ndvi = np.zeros((len(LN), days, 1))
    for i in range(days):
        day = start + timedelta(days=i)
        ndvi[:, i, 0] = modis_ndvi(day, AOI_lat, AOI_lon, search_size, LN)
    cwd = os.getcwd()
    ndvi_path = os.path.join(cwd, 'Data', 'ndvi3b.npy')
    np.save(ndvi_path, ndvi)  # save data


def ndvi_plot():
    NDVI = modis_ndvi(day, AOI_lat, AOI_lon, search_size, LN)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    LN.plot(ax=ax, column=np.array(NDVI), legend=True, cmap=cmocean.cm.rain, linewidth=3)
    plt.show()


if __name__ == "__main__":
    # show an illustrative plot of ndvi data on LN
    start = timeit.default_timer()
    day = date(2019, 6, 30)
    AOI_lat = [36, 37]
    AOI_lon = [-120, -119]
    search_size = 0.002  # set AOI and search size
    LN = config.LN
    ndvi_plot()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    start_time = timeit.default_timer()
    # obtain ndvi from modis data
    start = date(2019, 5, 29)
    end = date(2019, 7, 7)
    days = (end - start).days + 1  # set interested dates
    LN = config.LN  # input global variable
    ndvi_data()
    stop_time = timeit.default_timer()
    print('Time: ', stop_time - start_time)
