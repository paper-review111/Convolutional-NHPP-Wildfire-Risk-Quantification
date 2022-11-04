import os
import geopandas as gpd
import pandas as pd


cwd = os.getcwd()

LN_AOI_path = os.path.join(cwd, 'data', 'LN_AOI', 'LN_AOI.shp')
LN_s_AOI_path = os.path.join(cwd, 'data', 'LN_AOI', 'LN_s_AOI.pkl')
Fire_AOI_prj_path = os.path.join(cwd, 'data', 'Fire_AOI_prj', 'Fire_AOI_prj.shp')

if os.path.exists(LN_AOI_path):
    LN = gpd.read_file(LN_AOI_path)

if os.path.exists(LN_s_AOI_path):
    LN_s = pd.read_pickle(LN_s_AOI_path)

if os.path.exists(Fire_AOI_prj_path):
    fire = gpd.read_file(Fire_AOI_prj_path)

