import pandas as pd
import numpy as np
import scipy as scipy
import xarray as xr
import verde as vd
# For projecting data
import pyproj
import time

# ====== Note on generating the CSV from .nc ======
# Converting a NetCDF (.nc) file to CSV/XYZ format using GMT's `grd2xyz` doesn't include a header.
# So we first create an empty file, manually add the header, then append the data:
#
#   touch global_spline_RF.csv
#   echo "Longitude,Latitude,Moho" >> global_spline_RF.csv
#   gmt grd2xyz global_moho.nc >> global_spline_RF.csv

# ====== Begining ======
#Loading only Lat, Long, Moho_km
# Define file path
path_to_data_file = 'Global_crust_Hk.csv'
rf_error = 3

# Load only required columns while ignoring commented lines
data_eq_all = pd.read_csv(path_to_data_file, comment='#', usecols=["Lat", "Long", "Moho_km"])

# Convert Moho_km to numeric, forcing non-numeric values to NaN
data_eq_all["Moho_km"] = pd.to_numeric(data_eq_all["Moho_km"], errors="coerce")

data_eq = data_eq_all
eq_info = data_eq
print (data_eq)

region = [-180, 180, -90, 90]  # Covers the whole world
#region = [-180, 180, -89.9, 89.9] #avoids the poles
print(region)
print()

# Load Crust1.0
path_to_data_file_moho = 'Crust1.0/crust.csv'

# Read the file again with the extracted header
data_raw = pd.read_csv(path_to_data_file_moho, sep=r'\s+') ## sep is used instead of delim_whitespace = true as it will be removed in latest pandas
data_moho = data_raw.dropna()

print (data_moho)
print()

# Ensure numeric types for latitude and longitude
data_moho["longitude"] = pd.to_numeric(data_moho["longitude"], errors="coerce")
data_moho["latitude"] = pd.to_numeric(data_moho["latitude"], errors="coerce")
data_eq["Long"] = pd.to_numeric(data_eq["Long"], errors="coerce")
data_eq["Lat"] = pd.to_numeric(data_eq["Lat"], errors="coerce")


## Spline Interpolation
data = data_eq
coordinates=(data.Long, data.Lat)
moho=data.Moho_km
print (data)
#print (moho)

# Projection, Coordinates, Region and Spacing
coordinates = (data.Long.values, data.Lat.values)
region = vd.get_region(coordinates)

# Use a Mercator projection for our Cartesian gridder
projection = pyproj.Proj(proj="merc", lat_ts=data.Lat.mean())
print(region)
print(coordinates)
print()

# The output grid spacing will 6 arc-minutes. If n/60 then x arc minutes
#spacing = 6 / 60
spacing = 1

#spline

# Record the start time
start_time = time.time()
# This spline will automatically perform cross-validation and search for the optimal parameter configuration.
#spline = vd.SplineCV(dampings=(1e-4, 1e-3))
spline = vd.SplineCV(dampings=(1e-5, 1e-3))

spline.fit(projection(*coordinates), data.Moho_km) # projection(*coordinates)

# Spline Score and Dampings: Use only when usinge SplineCV otherwise comment the Score and Damping Prints
# We can show the best R² score obtained in the cross-validation
#print("\nScore: {:.3f}".format(spline.scores_.max())) # only for SplineCV

# And then the best damping parameter that produced this high score.
#print("\nBest damping:", spline.damping_) # only for SplineCV

# Now we can create a geographic grid of air temperature by providing a
# projection function to the grid method and mask points that are too far from
# the observations
grid_full = spline.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names="moho",
)

grid = vd.distance_mask(
    coordinates, maxdist=3 * spacing * 111e3, grid=grid_full, projection=projection
)

# Record the end time
end_time = time.time()

# Calculate and print the execution time
print(f"Execution time: {end_time - start_time:.2f} seconds")
print()

## Grid into a dataframe
grid_df = grid.to_dataframe().reset_index()
grid_full_df = grid_full.to_dataframe().reset_index()
# Optional: drop NaNs if there's a distance mask applied
spline_df = grid_full_df.dropna(subset=["moho"]).reset_index(drop=True)

print (spline_df)

# Crust1.0 lat,long with no rf
# Extract coordinates and RF Moho values
rf_lat = data_eq.Lat.values
rf_lon = data_eq.Long.values
rf_moho = data_eq.Moho_km.values

rf_coords = set(zip(rf_lat, rf_lon))

grid_full_df_no_rf = data_moho[~data_moho[["latitude", "longitude"]].apply(tuple, axis=1).isin(rf_coords)].reset_index(drop=True)
#####
# e.g., if lat=10 & long=20 then tuple makes it (10, 20)
# isin(rf_coords) returns a boolean True where those coordinates are present and ~ inverts the boolean to False meaning where those are not present
# data_moho(2) the coordiantes from this df is used
# data_moho(1) the master df from which we get the filtered row based on conditions
# .reset_index(drop=True) : resets the old index ,i.e., from the master df
#####

#Get Spline Moho Values
lats = grid_full.latitude.values
lons = grid_full.longitude.values
moho = grid_full.moho.values

lon_grid, lat_grid = np.meshgrid(lons, lats)
points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()]) # Flatten the values using ravel
values = moho.ravel()

get_spline_moho = scipy.interpolate.NearestNDInterpolator(points, values)

target_lats = grid_full_df_no_rf["latitude"].values
target_lons = grid_full_df_no_rf["longitude"].values

spline_moho_values = get_spline_moho(target_lats, target_lons)

#projected_coords = projection(grid_full_df_no_rf["longitude"].values,grid_full_df_no_rf["latitude"].values)
#spline.predict

grid_full_df_no_rf["spline_moho"] = spline_moho_values


df_rf = data_eq[["Lat", "Long", "Moho_km"]].copy()
df_rf.columns = ["Latitude", "Longitude", "Moho"]

lat = grid_full_df_no_rf["latitude"].values
lon = grid_full_df_no_rf["longitude"].values
moho_crust = grid_full_df_no_rf["Moho"].values
spline_moho = grid_full_df_no_rf["spline_moho"].values

moho_final = np.where(abs(moho_crust - spline_moho) <= rf_error, spline_moho, moho_crust) ## (Condition, if true: execute, else: execute)

df_crust = pd.DataFrame({"Latitude": lat,"Longitude": lon,"Moho": moho_final})

df_final_moho = pd.concat([df_rf, df_crust], ignore_index=True)
print(df_final_moho)
print()


# df to xarray
#ds_all = xr.Dataset.from_dataframe(df_final_moho)
# Save to NetCDF
#ds_all.to_netcdf("weighted_moho_all.nc")
#df_moho = df_final_moho.copy()
# Load your dataframe (ensure df_moho is already defined)
#ds = df_moho.to_xarray()
#print (ds)

df = df_final_moho.copy()

# Define the target grid
lat_grid = np.arange(-90, 90.1, 0.5)
lon_grid = np.arange(-180, 180.1, 0.5)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# Create the interpolator object with NearestNDInterpolator
final_moho_func = scipy.interpolate.NearestNDInterpolator(list(zip(df['Longitude'], df['Latitude'])),df['Moho'])

# Evaluate on the grid (lon_mesh, lat_mesh)
grid_z = final_moho_func(lon_mesh, lat_mesh)


# Create DataArray (do not add attrs here yet)
data_array = xr.DataArray(
    grid_z,
    coords={'Latitude': lat_grid, 'Longitude': lon_grid},
    dims=['Latitude', 'Longitude'],
    name='Moho'
)

# Step 5: Wrap in Dataset
ds = data_array.to_dataset()

# Step 6: Add variable-level metadata (must be done after to_dataset)
ds['Moho'].attrs = {
    'units': 'km',
    'long_name': 'Moho Depth',
    'standard_name': 'moho_depth',
    'actual_range': [float(np.nanmin(grid_z)), float(np.nanmax(grid_z))],
    '_FillValue': np.nan,  # Important for GMT/PyGMT compatibility
    'coordinates': 'Latitude Longitude',  # Indicate that these coordinates correspond to lat/lon
}

# Step 7: Add coordinate variable metadata
ds['Latitude'].attrs = {
    'units': 'degrees_north',
    'long_name': 'Latitude',
    'axis': 'Y'
}
ds['Longitude'].attrs = {
    'units': 'degrees_east',
    'long_name': 'Longitude',
    'axis': 'X'
}

# Step 8: Add global metadata
ds.attrs = {
    'title': 'Gridded Moho Depth',
    'summary': 'Interpolated onto regular 0.5x0.5 lat-lon grid',
    #'Conventions': 'CF-1.6',
    'creator_name': 'Your Name or Organization',  # Add creator info
    'institution': 'Your Institution',  # Add institution info
    'source': 'Original dataset details (e.g., source of Moho data)',  # Add data source info
    'history': 'Data processed using NearestNDInterpolator method for interpolation',  # Add processing info
    'date_created': str(pd.to_datetime('today'))  # Add date of creation
}

# Step 9: Save as NetCDF
ds.to_netcdf('global_moho.nc', format='NETCDF4')