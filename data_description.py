import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
import contextily as cx
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
# subset CA
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
