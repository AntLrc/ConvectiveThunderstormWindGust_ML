import numpy as np
import pandas as pd
import xarray as xr
import os


def create_labels(input_dir, years, output_dir):
    """
    Creates the files with the labels for the experiments to use.
    The labels are the observed gusts at the stations.
    
    Parameters
    ----------
    input_dir : str
        The directory where the data is stored.
    years : list of str
        The years for which the labels are created.
    output_dir : str
        The directory where the labels are saved.
    """

    def get_stations(input_dir):
        """
        Collect the stations that have data for all months between April and October.
        
        Parameters
        ----------
        input_dir : str
            The directory where the data is stored.
            
        Returns
        -------
        stations : np.array
            The list of stations that have data for all months between April and October.
        """
        # Load the data
        stations = None
        for file in os.listdir(input_dir):
            # Check that it is an .nc file, and that the month are between April and October
            if file[-3:] != ".nc" or int(file[-5:-3]) < 4 or int(file[-5:-3]) > 10:
                continue
            # Getting rid of incomplete data
            tmp = xr.open_dataset(os.path.join(input_dir, file))
            tmp = tmp.dropna(
                dim="station", thresh=0.7 * len(tmp.time) * 2
            ).station.values
            stations = tmp if stations is None else np.intersect1d(stations, tmp)
        return stations

    stats = get_stations(input_dir)

    def get_dates(input_dir, stations):
        """
        Collect the dates that have data for all stations.
        
        Parameters
        ----------
        input_dir : str
            The directory where the data is stored.
        stations : np.array
            The list of stations that have data for all months between April and October.
        
        Returns
        -------
        dates : np.array
            The list of dates that have data for all stations.
        """
        # Load the data
        dates = None
        for file in os.listdir(input_dir):
            # Check that it is an .nc file, and that the month are between April and October
            if file[-3:] != ".nc" or int(file[-5:-3]) < 4 or int(file[-5:-3]) > 10:
                continue
            # Getting rid of incomplete data
            tmp = xr.open_dataset(os.path.join(input_dir, file))
            tmp = tmp.sel(station=stations).dropna(dim="time", how="any").time.values
            dates = tmp if dates is None else np.concatenate((dates, tmp))
        return dates

    dates = get_dates(input_dir, stats)

    # Load the data
    for year in years:
        # Less efficient, more simple, I don't care
        result = None
        files_in_dir = os.listdir(input_dir)
        files_in_dir.sort()
        for file in files_in_dir:
            # if file is like "yyyy**.nc"...
            remaining = file.split("_")[-1]
            file_year, file_month, file_type = (
                remaining[:4],
                remaining[4:6],
                remaining[6:],
            )
            if (
                file_year == year
                and file_type == ".nc"
                and int(file_month) >= 4
                and int(file_month) <= 10
            ):
                data = xr.open_dataset(os.path.join(input_dir, file))
                # Select only the stations that are in the stats list and the dates that are in the dates list
                data = data.sel(
                    station=np.intersect1d(stats, data.station.values),
                    time=np.intersect1d(dates, data.time.values),
                )
                result = (
                    data if result is None else xr.concat([result, data], dim="time")
                )
        # Save the result
        result.to_netcdf(os.path.join(output_dir, year + "_labels.nc"))


def intersect_dates(nn_preinput_dir, yearInt=None):
    """
    The ncdf files for labels and input may have different dates.
    This function will intersect the dates of the input and labels files.
    year is an optional argument if you want to intersect only one year.
    
    Parameters
    ----------
    nn_preinput_dir : str
        The directory where the data is stored.
    yearInt : int
        The year for which the dates are intersected.
    """
    files = os.listdir(nn_preinput_dir)
    files.sort()

    # The files are named yyyy_labels.nc and yyyy_input.nc

    # Keep only years for which we have both labels and input
    years = [file[:4] for file in files]
    for year in set(years):
        years.remove(year)

    for year in years:
        if yearInt is not None and year != yearInt:
            continue
        inputs = xr.open_dataset(os.path.join(nn_preinput_dir, year + "_Pangu.nc"))
        labels = xr.open_dataset(os.path.join(nn_preinput_dir, year + "_labels.nc"))
        baselines = xr.open_dataset(os.path.join(nn_preinput_dir, year + "_Baseline.nc"))

        # Get dates of inputs and labels
        dates = np.intersect1d(inputs.time.values, labels.time.values)
        dates = np.intersect1d(dates, baselines.time.values)

        # Get rid of incomplete cases

        inputs = inputs.sel(time=dates)
        labels = labels.sel(time=dates)
        baselines = baselines.sel(time=dates)

        inputs.to_netcdf(os.path.join(nn_preinput_dir, "new", year + "_Pangu.nc"))
        labels.to_netcdf(os.path.join(nn_preinput_dir, "new", year + "_labels.nc"))
        baselines.to_netcdf(os.path.join(nn_preinput_dir, "new", year + "_Baseline.nc"))

        inputs.close()
        labels.close()
        baselines.close()


def create_input(input_dir, years, output_dir):
    """
    Creates the input files for the experiments to use.
    The input files are the meteorological data at the stations.
    
    Parameters
    ----------
    input_dir : str
        The directory where the data is stored.
    years : list of str
        The years for which the input is created.
    output_dir : str
        The directory where the input is saved.
    """
    files = os.listdir(input_dir)
    for file in files:
        print(file)
        if file[-8:] != "Pangu.nc" or (years is not None and file[:4] not in years):
            continue
        data = xr.open_dataset(os.path.join(input_dir, file))
        data = complete_cases(data)
        data.to_netcdf(os.path.join(output_dir, file))
        data.close()


def create_baseline_input(input_dir, years, output_dir):
    """
    Creates the baseline input files for the experiments to use.
    The baseline input files are the ERA5 data at the stations, used
    as if they had been predicted by Pangu.
    
    Parameters
    ----------
    input_dir : str
        The directory where the data is stored.
    years : list of str
        The years for which the input is created.
    output_dir : str
        The directory where the input is saved.
    """
    files = os.listdir(input_dir)
    for file in files:
        print(file)
        if file[-8:] != "Pangu.nc" or (years is not None and file[:4] not in years):
            continue
        data = xr.open_dataset(os.path.join(input_dir, file))
        for var in data.data_vars:
            for lt in data[var].lead_time.values:
                data[var][data[var].lead_time == lt] = (
                    data[var][data[var].lead_time == 0].shift(time=lt).values
                )
        data = complete_cases(data)
        data.to_netcdf(os.path.join(output_dir, file[:-8] + "Baseline.nc"))
        data.close()


def complete_cases(data):
    """
    Get rid of incomplete cases.
    
    Parameters
    ----------
    data : xr.Dataset
        The dataset to clean.
    
    Returns
    -------
    data : xr.Dataset
        The cleaned dataset.
    """
    # Get rid of incomplete cases
    data = data.dropna(
        dim="time",
        how="any",
        subset=np.setdiff1d(
            np.array(data.data_vars), np.array(["CAPE", "CIN", "LCL", "LFC"])
        ),
    )
    return data


def adapt_input(
    input_dir,
    output_dir,
    years=None,
    input_suffix="Pangu.nc",
    output_suffix="Interpolated.nc",
):
    """
    Uses the input created for the Neural Network to prepare the input for R.
    In practice, interpolates the data on the coordinates of the stations.
    
    Parameters
    ----------
    input_dir : str
        The directory where the data is stored.
    output_dir : str
        The directory where the data is saved.
    years : list of str
        The years for which the input is created.
    input_suffix : str
        The suffix of the input files.
    output_suffix : str
        The suffix of the output files.
    """
    for file in os.listdir(input_dir):
        if not file.endswith(input_suffix) or (
            years is not None and file[:4] not in years
        ):
            continue
        data = xr.open_dataset(os.path.join(input_dir, file))

        # Selecting the coordinates of the stations. Using only one time step as the coordinates are the same for all time steps.
        labels = (
            xr.open_dataset(
                os.path.join(input_dir, file[: -len(input_suffix)] + "labels.nc")
            )
            .isel(time=0)
            .drop("time")
        )
        lons, lats = labels.longitude, labels.latitude
        # Create new file with data intersected on the coordinates of inputs.to_netcdf(os.path.join(outputDir, year + "_Pangu.nc"))labels
        # Fill the missing values with 0, for CAPE and CIN
        data.fillna(0.0).interp(lon=lons, lat=lats).to_netcdf(
            os.path.join(output_dir, file[: -len(input_suffix)] + output_suffix)
        )
        data.close()
        labels.close()


def gust_factor(wind, gust):
    """
    Computes the gust factor from the wind and gust speeds.
    
    Parameters
    ----------
    wind : xr.DataArray
        The wind speed.
    gust : xr.DataArray
        The gust speed.
    """
    return (gust / wind).mean()


def era5_baseline(
    input_dir,
    input_interpolation_dir,
    output_dir,
    years,
    input_suffix,
    output_suffix="ERA5.nc",
):
    """
    Creates the baseline input files for the experiments to use,
    using the ERA5 wind gust as observation.
    
    Parameters
    ----------
    input_dir : str
        The directory where the data is stored.
    input_interpolation_dir : str
        The directory where the interpolated data is stored.
    output_dir : str
        The directory where the data is saved.
    years : list of str
        The years for which the input is created.
    input_suffix : str
        The suffix of the input files.
    output_suffix : str
        The suffix of the output files.
    """
    for year in years:
        for file in os.listdir(input_dir):
            if not file.endswith(".nc"):
                print(file)
                continue
            data = xr.open_dataset(os.path.join(input_dir, file))
            if not year in data.time.dt.year:
                data.close()
                continue
            data = data.sel(time=str(year))
            # First, create a new data array with correct dates
            ds_interp = xr.open_dataset(
                os.path.join(input_interpolation_dir, str(year) + "_" + input_suffix)
            )
            np_vals = data.fg10.values
            np_vals = np_vals.reshape(-1, np_vals.shape[-2], np_vals.shape[-1])
            np_vals = np.repeat(
                np.expand_dims(np_vals, axis=0), repeats=len(ds_interp.lead_time), axis=0
            )
            da = xr.DataArray(
                data=np_vals,
                coords={
                    "lead_time": (["lead_time"], ds_interp.lead_time.values),
                    "time": (["time"], data.fg10.valid_time.values.reshape(-1)),
                    "latitude": (["latitude"], data.latitude.values),
                    "longitude": (["longitude"], data.longitude.values),
                },
                name="ERA5_gust",
            )
            # Then, interpolate it on the coordinates of the stations
            lons, lats = ds_interp.longitude, ds_interp.latitude
            times = ds_interp.time
            da = da.sel(time=times).interp(longitude=lons, latitude=lats)
            ds = da.to_dataset()
            ds.to_netcdf(os.path.join(output_dir, str(year) + "_" + output_suffix))
            data.close()
            ds_interp.close()
            ds.close()


def add_persistant_vars(
    input_dir,
    output_dir,
    years=None,
    input_file_suffix="Pangu.nc",
    input_label_suffix="labels.nc",
    ouput_suffix="Pangu.nc",
):
    """
    Adds the gust at time - lead_time to the input data.
    
    Parameters
    ----------
    input_dir : str
        The directory where the data is stored.
    output_dir : str
        The directory where the data is saved.
    years : list of str
        The years for which the input is created.
    input_file_suffix : str
        The suffix of the input files.
    input_label_suffix : str
        The suffix of the label files.
    """
    files = os.listdir(input_dir)
    files.sort()
    # The files are named yyyy_labels.nc and yyyy_input.nc
    # Keep only years for which we have both labels and input
    file_years = [file[:4] for file in files]
    for year in set(file_years):
        file_years.remove(year)
    for year in file_years:
        if years is not None and not year in years and not int(year) in years:
            continue
        inputs = xr.open_dataset(os.path.join(input_dir, year + "_" + input_file_suffix))
        labels = xr.open_dataset(os.path.join(input_dir, year + "_" + input_label_suffix))
        for var in labels.data_vars:
            # Create var
            da = xr.DataArray(
                data=np.repeat(
                    np.expand_dims(labels[var].values, axis=0),
                    len(inputs.lead_time),
                    axis=0,
                ),
                dims=("lead_time", "time", "station"),
                coords={
                    "lead_time": inputs.lead_time,
                    "time": labels.time,
                    "station": labels.station,
                },
            )
            # Shift vars to make it persistant
            for lt in da.lead_time.values:
                da[da.lead_time == lt] = da[da.lead_time == 0].shift(time=lt).values
            inputs[var] = da.sel(time=inputs.time)
        inputs.to_netcdf(os.path.join(output_dir, year + "_" + ouput_suffix))
        inputs.close()
        labels.close()
