import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
import contextily as cx
import pandas as pd
from shapely.ops import nearest_points
import datetime as dt
from shapely.geometry import Polygon
import config
import seaborn as sns
import numpy as np
import cmocean
import warnings
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------------------------------------------------
# read the transmission lines data
# -----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd()
PG_CA_path = os.path.join(cwd, 'data', 'transmission_lines', 'transmission_lines.shp')
PG_CA = gpd.read_file(PG_CA_path)

# -----------------------------------------------------------------------------------------------------------------------
# plot the transmission lines in CA
# -----------------------------------------------------------------------------------------------------------------------
US_path = os.path.join(cwd, 'data', 'us_states.json')
US = gpd.read_file(US_path)
CA = US[US.id == 'CA']

fig1, ax = plt.subplots(1, figsize=(10, 10))
PG_CA.plot(ax=ax, color='black', linewidth=1)
CA.plot(ax=ax, facecolor='grey', alpha=0.3)
cx.add_basemap(ax, crs=PG_CA.crs)
fig1.patch.set_visible(False)
plt.xlabel('long', fontsize=26, color='black')
plt.ylabel('lat', fontsize=26, color='black')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
ax.add_patch(Rectangle((-120, 36), 1, 1, edgecolor='red', fill=False, lw=2.5))
plot_path = os.path.join(cwd, 'results', 'CA_transmission.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# plot the transmission line for Selected Area
# -----------------------------------------------------------------------------------------------------------------------
AOI_lat = [36, 36, 37, 37]
AOI_lon = [-119, -120, -120, -119]
AOI_geom = Polygon(zip(AOI_lon, AOI_lat))
AOI = gpd.GeoDataFrame(index=[0], crs=PG_CA.crs, geometry=[AOI_geom])
PG_AOI = gpd.overlay(PG_CA, AOI, how="intersection")

fig2, ax = plt.subplots(1, figsize=(10, 10))
PG_AOI.plot(ax=ax, color='black', linewidth=1)
cx.add_basemap(ax, crs=PG_AOI.crs)
fig2.patch.set_visible(False)
plt.xlabel('long', fontsize=26, color='black')
plt.ylabel('lat', fontsize=26, color='black')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(3)
    ax.spines[axis].set_color('red')
plot_path = os.path.join(cwd, 'results', 'AOI_transmission.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# plot the length of AOI segments
# -----------------------------------------------------------------------------------------------------------------------
LN_AOI = config.LN
LN_AOI_Cartesian = LN_AOI.to_crs('EPSG:5234')
length = LN_AOI_Cartesian.geometry.length
fig3, ax = plt.subplots(figsize=(10, 10))
ax = sns.histplot(data=length, kde=False, color='black')
plt.ylim(0, 1000)
plt.xlim(0, 4000)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
plt.xlabel('segment length (m)', fontsize=26, color='black')
plt.ylabel('count', fontsize=26, color='black')
plot_path = os.path.join(cwd, 'results', 'length.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# plot NDVI data
# -----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd()
modis_ft_path = os.path.join(cwd, 'data', 'ndvi.npy')  # load features from modis
modis_ft = np.load(modis_ft_path)[:, 5, 0]
LN_AOI = config.LN

fig4, ax = plt.subplots(1, figsize=(10, 10))
LN_AOI.plot(ax=ax, column=modis_ft, legend=True, linewidth=3, cmap=cmocean.cm.rain)
plt.xlabel('long', fontsize=26, color='black')
plt.ylabel('lat', fontsize=26, color='black')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
cb_ax = fig4.axes[1]
cb_ax.tick_params(labelsize=18)
plot_path = os.path.join(cwd, 'results', 'NDVI.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

fig5, ax = plt.subplots(figsize=(10, 10))
ax = sns.distplot(a=modis_ft, color='darkgreen', axlabel='NDVI')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
plt.xlabel('nvdi', fontsize=26, color='black')
plt.ylabel('density', fontsize=26, color='black')
plot_path = os.path.join(cwd, 'results', 'NDVIdensity.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------------------------------------------------------------------------------------
# plot the motivating example figure
# -----------------------------------------------------------------------------------------------------------------------
# PG_CA
cwd = os.getcwd()
PowerGrid_path = os.path.join(cwd, 'data', 'transmission_lines', 'transmission_lines.shp')
PowerGrid_raw = gpd.read_file(PowerGrid_path)
PowerGrid = PowerGrid_raw[['geometry']]

# select CA
US_path = os.path.join(cwd, 'data', 'us_states.json')
US = gpd.read_file(US_path)
AOI = US[US.id == 'CA']
PG_CA = gpd.overlay(PowerGrid, AOI, how="intersection")

# Fire_CA in year 2019
fire_path = os.path.join(cwd, 'data', 'fire_incident.csv')
fire_raw = pd.read_csv(fire_path, encoding='ISO-8859-1')
fire_raw['Longitude'] = pd.to_numeric(fire_raw['Longitude'], errors='coerce')
fire_raw['Latitude'] = pd.to_numeric(fire_raw['Latitude'], errors='coerce')
fire_raw = fire_raw.dropna(subset=['Longitude', 'Latitude']).reset_index()
fire_raw['Longitude'].isnull().values.any()
fire = gpd.GeoDataFrame(
    fire_raw, crs=PG_CA.crs,
    geometry=gpd.points_from_xy(fire_raw.Longitude, fire_raw.Latitude)
)
fire_date = []
for i in np.arange(len(fire)):
    fire_date.append(dt.datetime.strptime(fire.Date[i], "%m/%d/%Y"))
fire.Date = fire_date
fire = fire[fire.Year == 2019]
Fire_CA = gpd.overlay(fire, AOI, how="intersection")  # select fire incidences for AOI

# plot the motivating example figure
fig6, ax = plt.subplots(1, figsize=(10, 10))
ax.set_title('2019 Wildfires', fontsize=26)
ax.set_xlim([-124.5, -114])
ax.set_ylim([32, 42.5])
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
PG_CA.plot(ax=ax, color="black")
AOI.plot(ax=ax, color="grey", alpha=0.3)
Fire_CA.plot(ax=ax, color="red", markersize=40.0, alpha=0.6)
cx.add_basemap(ax, crs=PowerGrid.crs)
plt.xlabel('long', fontsize=26, color='black')
plt.ylabel('lat', fontsize=26, color='black')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)
plot_path = os.path.join(cwd, 'results', 'motivating_example.png')
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
