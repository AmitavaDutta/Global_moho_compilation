import pandas as pd
import numpy as np
import scipy as sc
import xarray as xr
import verde as vd
# For projecting data
import pyproj
import time

#Loading only Lat, Long, Moho_km
# Define file path
#path_to_data_file = "\Amitava_Laptop\Geodynamics-Project-IISERP\India_Tibet\Global_crust.csv" ## for Linux
#path_to_data_file = r"D:\Amitava\Projects\Spline_Moho\Global_moho_compilation\Global_crust.csv"
path_to_data_file = r"D:\Amitava\Projects\Spline_Moho\Global_moho_compilation\Global_crust_Hk.csv" # dataset with only Hk

# Load only required columns while ignoring commented lines
data_eq_all = pd.read_csv(path_to_data_file, comment='#', usecols=["Lat", "Long", "Moho_km"])

# Convert Moho_km to numeric, forcing non-numeric values to NaN
data_eq_all["Moho_km"] = pd.to_numeric(data_eq_all["Moho_km"], errors="coerce")

# Filter data based on longitude and latitude
data_eq_ind = data_eq_all[
    (data_eq_all["Long"] >= 65) & (data_eq_all["Long"] <= 110) & 
    (data_eq_all["Lat"] >= 5) & (data_eq_all["Lat"] <= 45)
].reset_index(drop=True)

data_eq = data_eq_all
eq_info = data_eq
print (data_eq)

region = [-180, 180, -90, 90]  # Covers the whole world
#region = [-180, 180, -89.9, 89.9] #avoids the poles
print(region)
print()

# Load Crust1.0
#path_to_data_file_moho = "/home/amitava/Geodynamics-Project-IISERP/tomo/crust_ind.csv" ## for Ubuntu
#path_to_data_file_moho = r"D:\Amitava_Laptop\Geodynamics-Project-IISERP\tomo\crust.csv"
path_to_data_file_moho = "D:\\Amitava\\Projects\\Spline_Moho\\Global_moho_compilation\\Crust1.0\\crust.csv"
#path_to_data_file_moho = "/home/amitava/Geodynamics-Project-IISERP/India_Tibet/RF_India/Crustal_thickness.csv"
#path_to_data_file = "/home/amitava/Geodynamics-Project-IISERP/India_Tibet/RF_India/TEC26522-mmc2_aug.csv"

# Read the file again with the extracted header
data_raw = pd.read_csv(path_to_data_file_moho, sep=r'\s+') ## sep is used instead of delim_whitespace = true as it will be removed in latest pandas
data_moho_all = data_raw.dropna()

# Apply the filtering criteria
data_moho_ind = data_moho_all[
    (data_moho_all["longitude"] >= 65) & (data_moho_all["longitude"] <= 110) & 
    (data_moho_all["latitude"] >= 5) & (data_moho_all["latitude"] <= 45)
].reset_index(drop=True)

data_moho = data_moho_all

crust1_moho = data_moho
#print (crust1_moho)
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

# Extract coordinates and RF Moho values
rf_lat = data_eq.Lat.values
rf_lon = data_eq.Long.values
rf_moho = data_eq.Moho_km.values

# Create a new DataFrame with latitude, longitude, RF Moho, Spline Moho, and pointwise variance
df = pd.DataFrame({
    "latitude": rf_lat,
    "longitude": rf_lon,
    "rf_moho": rf_moho,
    
})

# Print summary
print(df)



# Create KDTree for RF locations (df is assumed to contain RF-based values)
#rf_tree = cKDTree(list(zip(df.longitude, df.latitude)))

# Assign RF Moho directly where RF exists (100% weight to RF)
df["weighted_moho"] = df["rf_moho"]
df["source"] = "RF"

# Create 1°x1° bins to find bins with no RF data
df["lat_bin"] = (df["latitude"] // 1)
df["lon_bin"] = (df["longitude"] // 1)
data_moho["lat_bin"] = (data_moho["latitude"] // 1)
data_moho["lon_bin"] = (data_moho["longitude"] // 1)

# Identify bins without RF data
rf_bins = set(zip(df["lat_bin"], df["lon_bin"]))
all_bins = set(zip(data_moho["lat_bin"], data_moho["lon_bin"]))
missing_rf_bins = all_bins - rf_bins

# Subset data_moho where RF data is missing
df_no_rf = data_moho[data_moho[["lat_bin", "lon_bin"]].apply(tuple, axis=1).isin(missing_rf_bins)].copy()
df_no_rf["weighted_moho"] = df_no_rf["Moho"]
df_no_rf["source"] = "Crust1.0"  # default source

# KDTree for nearest spline_df points
#spline_tree = cKDTree(list(zip(spline_df.longitude, spline_df.latitude)))
#_, nearest_spline_idx = spline_tree.query(list(zip(df_no_rf.longitude, df_no_rf.latitude)))

# Get the nearest spline Moho values from precomputed spline_df
#df_no_rf["nearest_rf_spline_moho"] = spline_df.moho.iloc[nearest_spline_idx].values

# Nearest-neighbor interpolator for spline Moho values
nn_interp = sc.interpolate.NearestNDInterpolator(list(zip(spline_df.longitude, spline_df.latitude)),spline_df["moho"])

# Apply to df_no_rf coordinates
df_no_rf["nearest_rf_spline_moho"] = nn_interp(df_no_rf.longitude.values, df_no_rf.latitude.values)

# Compute difference
x = df_no_rf["nearest_rf_spline_moho"] - df_no_rf["weighted_moho"]
y = abs(x)

# Apply weighting logic with fixed 1.5 km threshold
updated_mask = y <= 1.5
df_no_rf.loc[updated_mask, "weighted_moho"] = (
    0.6 * df_no_rf.loc[updated_mask, "nearest_rf_spline_moho"] +
    0.4 * df_no_rf.loc[updated_mask, "weighted_moho"]
)
df_no_rf.loc[updated_mask, "source"] = "weighted spline and Crust1.0"

# Combine back with original RF-based data
df_final = pd.concat([df, df_no_rf], ignore_index=True)

# Rename columns if needed
df_final.rename(columns={"Lat": "latitude", "Long": "longitude"}, inplace=True)

# Save to CSV
df_final.to_csv("weighted_moho_with_source.csv", index=False)

# Output
print(df_final)

print()


# df to xarray
ds_all = xr.Dataset.from_dataframe(df_final)
# Save to NetCDF
#ds_all.to_netcdf("weighted_moho_all.nc")

#df_moho_depth = df_final[["latitude", "longitude", "weighted_moho"]]
## Rename the final data
df_moho_depth = df_final[["latitude", "longitude", "weighted_moho"]].rename(
    columns={
        "latitude": "Latitude",
        "longitude": "Longitude",
        "weighted_moho": "Moho"
    }
)

print (df_moho_depth)

df_moho = df_moho_depth.copy()

ds = df_moho.to_xarray()
print (ds)

# Step 1: Load your dataframe (ensure df_moho is already defined)
df = df_moho.copy()

# Step 2: Define the target grid
lat_grid = np.arange(-90, 90.1, 0.5)
lon_grid = np.arange(-180, 180.1, 0.5)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

'''
# Step 3: Interpolate the data with griddata
grid_z = griddata(
    points=(df['Longitude'], df['Latitude']),
    values=df['Moho'],
    xi=(lon_mesh, lat_mesh),
    method='linear'
)
'''
# Create the interpolator object with NearestNDInterpolator
final_moho_func = sc.interpolate.NearestNDInterpolator(list(zip(df['Longitude'], df['Latitude'])),df['Moho'])

# Evaluate on the grid (lon_mesh, lat_mesh)
grid_z = final_moho_func(lon_mesh, lat_mesh)


# Step 4: Create DataArray (do not add attrs here yet)
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
