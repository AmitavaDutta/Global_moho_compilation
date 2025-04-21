import pygmt
import pandas as pd
import numpy as np
import scipy
from scipy.spatial import cKDTree
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import xarray as xr
import verde as vd
# For projecting data
import pyproj
# For fetching sample datasets
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm, PowerNorm

### To Load the entire dataset
'''
#path_to_data_file = ensaio.fetch_alps_gps(version=1)
#print(path_to_data_file)
#path_to_data_file = "/home/amitava/Geodynamics-Project-IISERP/India_Tibet/RF_India/Ind_RF.csv" ## For Ubuntu
path_to_data_file = "D:\\Amitava_Laptop\\Geodynamics-Project-IISERP\\India_Tibet\\Global_crust.csv" ## For Windows OR use '\\' instead of '\' 
data_eq_raw = pd.read_csv(path_to_data_file, comment='#') # data_eq_raw = pd.read_csv(path_to_data_file, comment='#', delimiter=',')  ## No delimeter is defaulted as comma


########################
## Read the file to find the header line
#with open(path_to_data_file, 'r') as f:
#    for line in f:
#        if line.startswith('#'):
#            header = line.strip('#').strip().split()  # Remove '#' and split into column names
#            break
########################

# Convert Moho_km to numeric, forcing non-numeric values to NaN
data_eq_raw["Moho_km"] = pd.to_numeric(data_eq_raw["Moho_km"], errors="coerce")

# Drop rows with NaN values in Moho_km
data_eq_all = data_eq_raw.dropna(subset=["Moho_km"])

# df = df.reset_index(drop=True) # To reset index
data_eq_all = data_eq_all.reset_index(drop=True)


# Filter data based on longitude and latitude
data_eq_ind = data_eq_all[
    (data_eq_all["Long"] >= 65) & (data_eq_all["Long"] <= 110) & 
    (data_eq_all["Lat"] >= 5) & (data_eq_all["Lat"] <= 45)
].reset_index(drop=True)

#data_eq = data_eq_all

# Display the data

#eq_info = data_eq
data_eq_all
#data_eq.to_csv("eq_data.csv", index=False)
'''



# Loading only Lat, Long, Moho_km
# Define file path
#path_to_data_file = "\Amitava_Laptop\Geodynamics-Project-IISERP\India_Tibet\Global_crust.csv" ## For Linux
path_to_data_file = "D:\\Amitava\\Projects\\Spline_Moho\\Global_moho_compilation\\Global_crust.csv"   ## For Windows

# Load only required columns while ignoring commented lines
data_eq_raw = pd.read_csv(path_to_data_file, comment='#', usecols=["Lat", "Long", "Moho_km"])

# Convert Moho_km to numeric, forcing non-numeric values to NaN
data_eq_raw["Moho_km"] = pd.to_numeric(data_eq_raw["Moho_km"], errors="coerce")

# Drop rows with NaN values in Moho_km
data_eq_all = data_eq_raw.dropna(subset=["Moho_km"]).reset_index(drop=True)

# Filter data based on longitude and latitude
data_eq_ind = data_eq_all[
    (data_eq_all["Long"] >= 65) & (data_eq_all["Long"] <= 110) & 
    (data_eq_all["Lat"] >= 5) & (data_eq_all["Lat"] <= 45)
].reset_index(drop=True)

# Check for duplicate (Lat, Long, Moho_km) groups
duplicate_mask = data_eq_all.duplicated(subset=["Lat", "Long", "Moho_km"], keep=False)

# Count duplicate entries
num_duplicates = duplicate_mask.sum()
# Count unique groups that are duplicated
num_repeated_groups = data_eq_all.loc[duplicate_mask, ["Lat", "Long", "Moho_km"]].drop_duplicates().shape[0]

print(f"Total number of repeated (Lat, Long, Moho_km) rows: {num_duplicates}")
print(f"Number of unique (Lat, Long, Moho_km) groups that are repeated: {num_repeated_groups}")

# Keep only the first occurrence of each duplicate group
data_eq = data_eq_all.drop_duplicates(subset=["Lat", "Long", "Moho_km"], keep="first").reset_index(drop=True)

# Display the final DataFrame
eq_info = data_eq
print (data_eq)
print()

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

# Apply the filtering criteria
data_moho_ind = data_raw[
    (data_raw["longitude"] >= 65) & (data_raw["longitude"] <= 110) & 
    (data_raw["latitude"] >= 5) & (data_raw["latitude"] <= 45)
].reset_index(drop=True)

data_moho = data_raw.dropna()
#data_moho = data_moho_ind.dropna()
#print(data.head())
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
import time

# Record the start time
start_time = time.time()
# Your entire script here
# This spline will automatically perform cross-validation and search for the
# optimal parameter configuration.
#spline = vd.SplineCV(dampings=(1e-5, 1e-3, 1e-1)) #std = 4.6
#spline = vd.SplineCV(dampings=(1e-7, 1e-5, 1e-3, 1e-1))
spline = vd.SplineCV(dampings=(1e-5, 1e-3))
#spline = vd.Spline(damping=1e-7)

# Fit the model on the data. Under the hood, the class will perform K-fold
# cross-validation for each the 3 parameter values and pick the one with the
# highest score.
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

# Weight Distribution
## Standard Deviation of Spline wrt RF

# Extract coordinates and RF Moho values
rf_lat = data_eq.Lat.values
rf_lon = data_eq.Long.values
rf_moho = data_eq.Moho_km.values

# Interpolate the spline Moho values at RF locations
spline_moho = spline.predict(projection(rf_lon, rf_lat))


# Compute the overall standard deviation (STD) of the residuals (RF Moho - Spline Moho)
overall_std = np.std(rf_moho - spline_moho)

# Compute pointwise variance (squared error) for each data point
pointwise_variance = abs((rf_moho - spline_moho)) #** 2

# Create a new DataFrame with latitude, longitude, RF Moho, Spline Moho, and pointwise variance
df = pd.DataFrame({
    "latitude": rf_lat,
    "longitude": rf_lon,
    "rf_moho": rf_moho,
    "spline_moho": spline_moho,
    "err_moho": pointwise_variance  # Variance (squared error)
})

# Print summary
print(df)
print("\nOverall STD of Spline relative to RF Moho:", overall_std)
#df.to_csv("spline_data.csv", index=False)
print()

# Create a KDTree for RF locations
rf_tree = cKDTree(list(zip(df.longitude, df.latitude)))

# Assign RF Moho directly where RF exists (100% weight to RF)
df["weighted_moho"] = df["rf_moho"]

# Create 1°x1° bins and check which bins have no RF data
df["lat_bin"] = (df["latitude"] // 1) * 1
df["lon_bin"] = (df["longitude"] // 1) * 1

data_moho["lat_bin"] = (data_moho["latitude"] // 1) * 1
data_moho["lon_bin"] = (data_moho["longitude"] // 1) * 1

rf_bins = set(zip(df["lat_bin"], df["lon_bin"]))
all_bins = set(zip(data_moho["lat_bin"], data_moho["lon_bin"]))

# Identify bins with no RF data
missing_rf_bins = all_bins - rf_bins
df_no_rf = data_moho[data_moho[["lat_bin", "lon_bin"]].apply(tuple, axis=1).isin(missing_rf_bins)].copy()

# Assign Crust1.0 Moho first
df_no_rf["weighted_moho"] = df_no_rf["Moho"]


# Find the nearest RF node for each location without RF
_, nearest_rf_idx = rf_tree.query(list(zip(df_no_rf.longitude, df_no_rf.latitude)))

# Get **precomputed** spline_moho from the nearest RF node
df_no_rf["nearest_rf_spline_moho"] = df.spline_moho.iloc[nearest_rf_idx].values

# Predict spline Moho for each missing RF location using the global spline
df_no_rf["nearest_rf_spline_moho"] = spline.predict(
    projection(df_no_rf.longitude, df_no_rf.latitude)
)

# Compute x and y using the **precomputed nearest RF spline Moho**
x = df_no_rf["nearest_rf_spline_moho"] - df_no_rf["weighted_moho"]
y = abs(x)

# Apply updated weighting logic
df_no_rf["weighted_moho"] = np.where(
    y < overall_std,  
    df_no_rf["nearest_rf_spline_moho"],  # If error is small, use nearest RF spline Moho
    0.4 * df_no_rf["nearest_rf_spline_moho"] + 0.6 * df_no_rf["weighted_moho"]  # Otherwise, use weighted combination
)

# Combine both datasets
df_final = pd.concat([df, df_no_rf], ignore_index=True)

# Save to CSV
df_final.to_csv("weighted_moho.csv", index=False)

# Print summary
print(df_final)
print()


# df to xarray
ds_all = xr.Dataset.from_dataframe(df_final)
# Save to NetCDF
#ds_all.to_netcdf("weighted_moho_all.nc")

# Select only the desired columns
df_subset = df_final[["latitude", "longitude", "weighted_moho"]]

# Convert to xarray Dataset
ds = xr.Dataset.from_dataframe(df_subset)

# Save to NetCDF
ds.to_netcdf("weighted_moho.nc")

grid.to_netcdf("spline_ds.nc")

print(ds)
print('Weighted moho dataset written on weighted_moho.nc and the spline written on spline_ds.nc')