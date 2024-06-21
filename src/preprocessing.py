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
        
        # Delete old files
        # os.remove(os.path.join(NNpreinputDir, year + "_Pangu.nc"))
        # os.remove(os.path.join(NNpreinputDir, year + "_labels.nc"))
        
        # Rename new files
        # os.rename(os.path.join(NNpreinputDir, "B" + year + "_Pangu.nc"), os.path.join(NNpreinputDir, year + "_Pangu.nc"))
        # os.rename(os.path.join(NNpreinputDir, "B" + year + "_labels.nc"), os.path.join(NNpreinputDir, year + "_labels.nc"))
        
        # I have no other choice because I cannot modify nc file in place
        
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
    
    



