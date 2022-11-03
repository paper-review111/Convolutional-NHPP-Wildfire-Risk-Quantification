import os
import geopandas as gpd
import datetime as dt
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from util import LineCutFunc, DayCalFunc
from itertools import chain
import pandas as pd
import numpy as np
from pyproj import Geod
import timeit
import warnings

warnings.filterwarnings('ignore')

start = timeit.default_timer()
# -----------------------------------------------------------------------------------------------------------------------
# read the transmission line data from EIA
# -----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd()
PowerGrid_path = os.path.join(cwd, 'data', 'transmission_lines', 'transmission_lines.shp')
PowerGrid_raw = gpd.read_file(PowerGrid_path)
PowerGrid = PowerGrid_raw[['geometry']]

# -----------------------------------------------------------------------------------------------------------------------
# select an interested area (AOI)
# -----------------------------------------------------------------------------------------------------------------------
# set CA as AOI
# US_path = os.path.join(cwd, 'data', 'us_states.json')
# US = gpd.read_file(US_path)
# AOI = US[US.id == 'CA']

# create a customized AOI
AOI_lat = [36, 36, 37, 37]
AOI_lon = [-119, -120, -120, -119]
AOI_geom = Polygon(zip(AOI_lon, AOI_lat))
AOI = gpd.GeoDataFrame(index=[0], crs=PowerGrid.crs, geometry=[AOI_geom])

PG_AOI = gpd.overlay(PowerGrid, AOI, how="intersection")

# -----------------------------------------------------------------------------------------------------------------------
# plot the transmission line for AOI
# -----------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, figsize=(20, 20))
PG_AOI.plot(ax=ax, color='black', linewidth=3)
AOI.plot(ax=ax, facecolor='grey', alpha=0.0)
cx.add_basemap(ax, crs=PowerGrid.crs)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# create a LinearNetwork for AOI
# -----------------------------------------------------------------------------------------------------------------------
n_seg = len(PG_AOI.geometry)
LinearNetwork = []
for i in range(n_seg):
    tempt = LineCutFunc(PG_AOI.geometry.geometry[i])
    LinearNetwork.append(tempt)
LinearNetwork = list(chain.from_iterable(LinearNetwork))
LN_AOI = gpd.GeoDataFrame(geometry=LinearNetwork, crs=PG_AOI.crs)

# add the center points v2
CP_Lat = []
CP_Lon = []
for i in range(len(LN_AOI)):
    CP = LN_AOI.geometry[i].centroid
    CP_Lat.append(CP.y)
    CP_Lon.append(CP.x)

# add the neighbor and their distance v2
wgs84_geod = Geod(ellps='WGS84')
nbor = []
nborDist = []
for i in range(len(LN_AOI)):
    line = LN_AOI.geometry[i]
    # find the neighbor id
    id = LN_AOI.geometry.intersects(line)
    id_nb = np.where(id)[0]
    nbor.append(id_nb)
    # find the distance of neighbor linestring
    Dist = []
    lat0, lon0 = (CP_Lat[i], CP_Lon[i])
    for j in np.arange(len(id_nb)):
        lat1, lon1 = (CP_Lat[id_nb[j]], CP_Lon[id_nb[j]])
        az12, az21, dist = wgs84_geod.inv(lon0, lat0, lon1, lat1)
        Dist.append(round(dist / 1000, 4))
    nborDist.append(Dist)
LN_s_AOI_data = {'CP_Lon': CP_Lon, 'CP_Lat': CP_Lat, 'NBor': nbor, 'NBorDist': nborDist}
LN_s_AOI = pd.DataFrame(data=LN_s_AOI_data)
# print(LN_s_AOI.NBor[1])
# print(type(LN_s_AOI.NBor[1]))
# print(LN_s_AOI.NBorDist[1])
# print(type(LN_s_AOI.NBorDist[1]))

# save the LN_AOI and LN_s
LN_AOI_path = os.path.join(cwd, 'data', 'LN_AOI', 'LN_AOI.shp')
LN_s_AOI_path = os.path.join(cwd, 'data', 'LN_AOI', 'LN_s_AOI.pkl')
LN_AOI.to_file(LN_AOI_path, index=False)
LN_s_AOI.to_pickle(LN_s_AOI_path)
# check the saved LN_AOI and LN_s data
# LN = gpd.read_file(LN_AOI_path)
# LN_s = pd.read_pickle(LN_s_AOI_path)
# print(LN_s.NBor[1])
# print(type(LN_s.NBor[1]))
# print(LN_s.NBorDist[1])
# print(type(LN_s.NBorDist[1]))

print(len(LN_AOI))

# -----------------------------------------------------------------------------------------------------------------------
# read fire incidences data
# -----------------------------------------------------------------------------------------------------------------------
fire_path = os.path.join(cwd, 'data', 'fire_incident.csv')
fire_raw = pd.read_csv(fire_path, encoding='ISO-8859-1')

fire_raw['Longitude'] = pd.to_numeric(fire_raw['Longitude'], errors='coerce')
fire_raw['Latitude'] = pd.to_numeric(fire_raw['Latitude'], errors='coerce')

fire_raw = fire_raw.dropna(subset=['Longitude', 'Latitude']).reset_index()
fire_raw['Longitude'].isnull().values.any()

fire = gpd.GeoDataFrame(
    fire_raw[['Date']], crs=LN_AOI.crs,
    geometry=gpd.points_from_xy(fire_raw.Longitude, fire_raw.Latitude)
)

fire_date = []
for i in np.arange(len(fire)):
    fire_date.append(dt.datetime.strptime(fire.Date[i], "%m/%d/%Y"))
fire.Date = fire_date

# -----------------------------------------------------------------------------------------------------------------------
# project fire locations to the transmission line for AOI
# -----------------------------------------------------------------------------------------------------------------------
# select fire incidences for AOI
Fire_AOI = gpd.overlay(fire, AOI, how="intersection")

# calculate the nearest distance between fire and transmission line
Fire_AOI_Cartesian = Fire_AOI.to_crs('EPSG:5234')
LN_AOI_Cartesian = LN_AOI.to_crs('EPSG:5234')
Fire_dist = []
LN_id_neast = []
for i in range(len(Fire_AOI)):
    dist = LN_AOI_Cartesian.geometry.distance(Fire_AOI_Cartesian.geometry[i])
    LN_id_neast.append(np.argmin(dist))
    Fire_dist.append(min(dist))

# add the nearest LN_id
Fire_AOI_Cartesian['LN_id'] = LN_id_neast

# # plot the Fire_dist
# a_bins = np.linspace(0, max(Fire_dist), num=20)
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.hist(Fire_dist, bins=a_bins)
# plt.show()

# find the nearest points on transmission line for Fire_AOI
for i in range(len(Fire_AOI_Cartesian)):
    p_prj, p = nearest_points(
        LN_AOI_Cartesian.geometry[LN_id_neast[i]], Fire_AOI_Cartesian.geometry[i]
    )
    Fire_AOI_Cartesian.geometry[i] = p_prj
Fire_AOI_prj = Fire_AOI_Cartesian.to_crs('EPSG:4326')

TOIs = '2019-6-1'
TOIe = '2019-6-30'
TOI = (pd.to_datetime(Fire_AOI_prj.Date) >= TOIs) & (pd.to_datetime(Fire_AOI_prj.Date) <= TOIe)
Fire_AOI_prj = Fire_AOI_prj[TOI].reset_index(drop=True)
Fire_AOI_prj['Date'] = Fire_AOI_prj.Date.apply(lambda x: str(x)[:10])
Fire_AOI_prj['Day'] = Fire_AOI_prj.Date.apply(DayCalFunc, starting_date=TOIs)

# save the Fire_AOI_prj
Fire_AOI_prj_path = os.path.join(cwd, 'data', 'fire_AOI_prj', 'Fire_AOI_prj.shp')
Fire_AOI_prj.to_file(Fire_AOI_prj_path)

print(Fire_AOI_prj)
stop = timeit.default_timer()
print('Time: ', stop - start)


def main():
    print('powerline_fire file runs successfully')


if __name__ == "__main__":
    main()
