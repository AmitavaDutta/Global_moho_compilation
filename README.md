# Crust

1) Crust_Map.ipynb : Original Jupyterlab source file, you can use this .ipynb file to check the maps generated with the splines, rfs, crust1.0 etc using cartopy and pygmt.
2) Global_crust.csv is the master file for crustal data and related informations compiled from a bunch of literature. This file is subject to further updates.
3) Global_crust_Hk.csv : This is the filtered moho data containing only moho from RF and removed duplicated using data_filter.ipynb
4) spline_crust.py : This is the final script used to generate the grid: global_moho.nc

### Updates Required: 
1) Refine the interpolation for generating the final grid from a dataframe(pd)/dataset(xr). [use interpolation used in V2RhoTgibbs]
2) Implement sediments into the origina datafiles to accounrt forsediment thickness.
3) Implement gravity calculation from tesseroid in harmonica from fatiando a terra.

