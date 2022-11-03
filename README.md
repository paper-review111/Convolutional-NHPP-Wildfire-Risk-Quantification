# Convolutional-NHPP-Wildfire-Risk-Quantification
## Environment
These codes are implemented in Python 3.10 with the following required packages. 
```html
cmocean 2.0
contextily 1.2.0
gemgis 1.0.0
geopandas 0.11.0
matplotlib 3.5.1 
numpy 1.22.3
pandas 1.4.2
pyproj 3.3.1
pytorch 1.13.0.dev20220729
requests 2.28.1
rioxarray 0.11.1
scipy 1.7.3 
seaborn 0.11.2
shapely 1.8.2
statsmodels 0.13.2
xarray 0.20.1
```
## Data 
- Power transmission lines data: https://www.eia.gov/maps/layer_info-m.php.
- Power-line wildfire incident data: https://www.cpuc.ca.gov/industries-and-topics/wildfires.
- Meteorological data from the NOAA HRRR model: https://console.cloud.google.com/marketplace/product/noaa-public/hrrr?project=python-232920&pli=1.
- Vegetation data from the NASA MODIS data product: https://ladsweb.modaps.eosdis.nasa.gov/search/order/4/MOD09GQ--61/2022-07-08..2022-07-09/DB/Tile:H8V5.
## Usage
- ```powerline_fire.py``` Before processing the environmental variables, the linear network of the power transmission lines should be made by [powerline_fire.py](https://github.com/paper-review111/Convolutional-NHPP-Wildfire-Risk-Quantification/blob/main/powerline_fire.py). In addiction, this file also projects the wildfire incident data to the linear network of the power transmisssion lines for the interested area.
- [noaa_hrrr.py](https://github.com/paper-review111/Convolutional-NHPP-Wildfire-Risk-Quantification-for-Power-Transmission-Lines/blob/main/noaa_hrrr.py) is used to obtain and process the NOAA-HRRR dataset. This file can download the dataset online and process the meteorological data for the power transmission lines. 
- [modis_ndvi.py](https://github.com/paper-review111/Convolutional-NHPP-Wildfire-Risk-Quantification-for-Power-Transmission-Lines/blob/main/modis_ndvi.py) are used to process the NOAA-HRRR and NASA-MODIS datasets respectively.
## Reproducibility
-
-
-
-
