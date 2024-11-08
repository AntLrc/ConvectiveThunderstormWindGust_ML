import numpy as np
import pandas as pd
import xarray as xr
import os

def createLabels(inputDir, years, outputDir):
    
    def getStations(inputDir):
        # Load the data
        stations = None
        for file in os.listdir(inputDir):
            # Check that it is an .nc file, and that the month are between April and October
            if file[-3:] != ".nc" or int(file[-5:-3]) < 4 or int(file[-5:-3]) > 10:
                continue
            # Getting rid of incomplete data
            tmp = xr.open_dataset(os.path.join(inputDir, file))
            tmp = tmp.dropna(dim='station', thresh = 0.7*len(tmp.time)*2).station.values
            stations = tmp if stations is None else np.intersect1d(stations, tmp)
        return stations
    
    stats = getStations(inputDir)
    
    def getDates(inputDir, stations):
        # Load the data
        dates = None
        for file in os.listdir(inputDir):
            # Check that it is an .nc file, and that the month are between April and October
            if file[-3:] != ".nc" or int(file[-5:-3]) < 4 or int(file[-5:-3]) > 10:
                continue
            # Getting rid of incomplete data
            tmp = xr.open_dataset(os.path.join(inputDir, file))
            tmp = tmp.sel(station = stations).dropna(dim='time', how = "any").time.values
            dates = tmp if dates is None else np.concatenate((dates, tmp))
        return dates
    
    dates = getDates(inputDir, stats)
    
    # Load the data
    for year in years:
        #Less efficient, more simple, I don't care
        result = None
        files_in_dir = os.listdir(inputDir)
        files_in_dir.sort()
        for file in files_in_dir:
            # if file is like "yyyy**.nc"...
            remaining = file.split("_")[-1]
            file_year, file_month, file_type = remaining[:4], remaining[4:6], remaining[6:]
            if file_year == year and file_type == ".nc" and int(file_month) >= 4 and int(file_month) <= 10: 
                data = xr.open_dataset(os.path.join(inputDir, file))
                # Select only the stations that are in the stats list and the dates that are in the dates list
                data = data.sel(station = np.intersect1d(stats, data.station.values),
                                time = np.intersect1d(dates, data.time.values))
                result = data if result is None else xr.concat([result, data], dim = "time")
        # Save the result
        result.to_netcdf(os.path.join(outputDir, year + "_labels.nc"))
                
def intersectDates(NNpreinputDir, yearInt = None):
    """
    The ncdf files for labels and input may have different dates.
    This function will intersect the dates of the input and labels files.
    year is an optional argument if you want to intersect only one year.
    """
    files = os.listdir(NNpreinputDir)
    files.sort()
    
    # The files are named yyyy_labels.nc and yyyy_input.nc
    
    # Keep only years for which we have both labels and input
    years = [file[:4] for file in files]
    for year in set(years):
        years.remove(year)
        
    for year in years:
        if yearInt is not None and year != yearInt:
            continue
        inputs = xr.open_dataset(os.path.join(NNpreinputDir, year + "_Pangu.nc"))
        labels = xr.open_dataset(os.path.join(NNpreinputDir, year + "_labels.nc"))
        baselines = xr.open_dataset(os.path.join(NNpreinputDir, year + "_Baseline.nc"))
        
        # Get dates of inputs and labels
        dates = np.intersect1d(inputs.time.values, labels.time.values)
        dates = np.intersect1d(dates, baselines.time.values)
        
        #Get rid of incomplete cases
        
        
        inputs = inputs.sel(time = dates)
        labels = labels.sel(time = dates)
        baselines = baselines.sel(time = dates)
        
        inputs.to_netcdf(os.path.join(NNpreinputDir, "new", year + "_Pangu.nc"))
        labels.to_netcdf(os.path.join(NNpreinputDir, "new", year + "_labels.nc"))
        baselines.to_netcdf(os.path.join(NNpreinputDir, "new", year + "_Baseline.nc"))
        
        inputs.close()
        labels.close()
        baselines.close()
        
def createInput(inputDir, years, outputDir):
    files = os.listdir(inputDir)
    for file in files:
        print(file)
        if file[-8:] != "Pangu.nc" or (years is not None and file[:4] not in years):
            continue
        data = xr.open_dataset(os.path.join(inputDir, file))
        data = completeCases(data)
        data.to_netcdf(os.path.join(outputDir, file))
        data.close()

def createBaselineInput(inputDir, years, outputDir):
    files = os.listdir(inputDir)
    for file in files:
        print(file)
        if file[-8:] != "Pangu.nc" or (years is not None and file[:4] not in years):
            continue
        data = xr.open_dataset(os.path.join(inputDir, file))
        for var in data.data_vars:
            for lt in data[var].lead_time.values:
                data[var][data[var].lead_time == lt] = data[var][data[var].lead_time == 0].shift(time = lt).values
        data = completeCases(data)
        data.to_netcdf(os.path.join(outputDir, file[:-8] + "Baseline.nc"))
        data.close()
            
def completeCases(data):
    # Get rid of incomplete cases
    data = data.dropna(dim='time',
                        how = "any",
                        subset = np.setdiff1d(np.array(data.data_vars), np.array(["CAPE", "CIN", "LCL", "LFC"])))
    return data
    
def adaptInput(inputDir, outputDir, years = None, filesuffix = "Pangu.nc", outputsuffix = "Interpolated.nc"):
    """
    Uses the input created for the Neural Network to prepare the input for R.
    In practice, interpolates the data on the coordinates of the stations.
    """
    for file in os.listdir(inputDir):
        if not file.endswith(filesuffix) or (years is not None and file[:4] not in years):
            continue
        data = xr.open_dataset(os.path.join(inputDir, file))
        
        # Selecting the coordinates of the stations. Using only one time step as the coordinates are the same for all time steps.
        labels = xr.open_dataset(os.path.join(inputDir, file[:-len(filesuffix)] + "labels.nc")).isel(time = 0).drop("time")
        lons, lats = labels.longitude, labels.latitude
        # Create new file with data intersected on the coordinates of inputs.to_netcdf(os.path.join(outputDir, year + "_Pangu.nc"))labels
        # Fill the missing values with 0, for CAPE and CIN
        data.fillna(0.).interp(lon = lons, lat = lats).to_netcdf(os.path.join(outputDir, file[:-len(filesuffix)] + outputsuffix))
        data.close()
        labels.close()

def gust_factor(wind, gust):
    """
    Computes the gust factor from the wind and gust speeds.
    """
    return (gust / wind).mean()

def ERA5baseline(inputDir, inputInterpolationDir, outputDir, years, interpsuffix, outputsuffix = "ERA5.nc"):
    """
    years should be a list of int
    """
    for year in years:
        for file in os.listdir(inputDir):
            if not file.endswith(".nc"):
                print(file)
                continue
            data = xr.open_dataset(os.path.join(inputDir, file))
            if not year in data.time.dt.year:
                data.close()
                continue
            data = data.sel(time = str(year))
            # First, create a new data array with correct dates
            dsinterp = xr.open_dataset(os.path.join(inputInterpolationDir, str(year) + "_" + interpsuffix))
            npvals = data.fg10.values
            npvals = npvals.reshape(-1, npvals.shape[-2], npvals.shape[-1])
            npvals = np.repeat(np.expand_dims(npvals, axis = 0), repeats = len(dsinterp.lead_time), axis = 0)
            da = xr.DataArray(
                data = npvals,
                coords = {"lead_time":(["lead_time"], dsinterp.lead_time.values),
                          "time":(["time"], data.fg10.valid_time.values.reshape(-1)),
                          "latitude":(["latitude"], data.latitude.values),
                          "longitude":(["longitude"], data.longitude.values)},
                name = "ERA5_gust"
            )
            # Then, interpolate it on the coordinates of the stations
            lons, lats = dsinterp.longitude, dsinterp.latitude
            times = dsinterp.time
            da = da.sel(time = times).interp(longitude = lons, latitude = lats)
            ds = da.to_dataset()
            ds.to_netcdf(os.path.join(outputDir, str(year) + "_" + outputsuffix))
            data.close()
            dsinterp.close()
            ds.close()
            
            
            

def add_persistant_vars(inputDir, outputDir, years = None, filesuffix = "Pangu.nc", labelsuffix = "labels.nc", outputsuffix = "Pangu.nc"):
    """
    Adds the gust at time - lead_time to the input data.
    """
    files = os.listdir(inputDir)
    files.sort()
    # The files are named yyyy_labels.nc and yyyy_input.nc
    # Keep only years for which we have both labels and input
    fileyears = [file[:4] for file in files]
    for year in set(fileyears):
        fileyears.remove(year)
    for year in fileyears:
        if years is not None and not year in years and not int(year) in years:
            continue
        inputs = xr.open_dataset(os.path.join(inputDir, year + "_" + filesuffix))
        labels = xr.open_dataset(os.path.join(inputDir, year + "_" + labelsuffix))
        for var in labels.data_vars:
            # Create var
            da = xr.DataArray(
                data = np.repeat(np.expand_dims(labels[var].values, axis = 0), len(inputs.lead_time), axis = 0),
                dims = ('lead_time', 'time', 'station'),
                coords = {'lead_time': inputs.lead_time,
                            'time': labels.time,
                            'station': labels.station},
            )
            # Shift vars to make it persistant
            for lt in da.lead_time.values:
                da[da.lead_time == lt] = da[da.lead_time == 0].shift(time = lt).values
            inputs[var] = da.sel(time = inputs.time)
        inputs.to_netcdf(os.path.join(outputDir, year + "_" + outputsuffix))
        inputs.close()
        labels.close()