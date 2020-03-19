#!/usr/bin/env python3
"""
Validate a PV forecast against PV_Live.

- Vlad Bondarenko <vbondarenko1@sheffield.ac.uk>
- Jamie Taylor <jamie.taylor@sheffield.ac.uk>
- First Authored: 2020-03-13
"""

import argparse
from datetime import timedelta, time
import pandas as pd
import numpy as np

from pvlive_api import PVLive
from helper import shift_n_days, load_from_file, to_utc

class Validation:
    def __init__(self, options=None):
        if options is None:
            self.options = self.parse_options()
        else:
            self.options = options

    def run_validation(self, forecast=None, min_yield=0.05):
        data = self.get_data(forecast, min_yield)
        horizons = data.index.get_level_values(1)
        dt = data.index.get_level_values(0)
        display_data = dict()
        display_data['data'] = dict()
        display_data['data']['Region0'] = dict()
        display_data['data']['Region0']['intraday'] = dict()
        display_data['data']['Region0']['day1'] = dict()
        for base in dt.hour.unique():
            base_str = str(time(hour=base))[:5]
            intraday = data[(dt.hour == base) & (horizons.isin(np.arange(0, 48 - base)))]
            dayp1 = data[(dt.hour == base) & (horizons.isin(np.arange(48 - (base * 2), 96 - (base * 2))))]
            preiod_names = ['intraday', 'day1']
            for i, period_data in enumerate([intraday, dayp1]):
                display_data['data']['Region0'][preiod_names[i]][base_str] = dict()
                display_period = display_data['data']['Region0'][preiod_names[i]][base_str]
                pred, actual, cap = period_data['forecast'], period_data['actual'], period_data['cap']
                pred_u, actual_u, cap_u = pred.unstack(), actual.unstack(), cap.unstack()
                errors = pd.DataFrame(index=pred_u.index)
                errors['mape'] = self.wmape(pred_u, actual_u, cap_u, axis=1)
                errors['r_squared'] = self.r_squared(pred_u, actual_u)
                display_period['r_squared'] = errors['r_squared'].values.tolist()
                display_period['mape'] = errors['mape'].values.tolist()
                display_period['actual'] = actual.values.tolist()
                display_period['predicted'] = pred.values.tolist()
                display_period['heatmap'] = self.calc_heatmap(pred, actual, cap)
        return display_data

    def get_data(self, forecast, min_yield):
        if forecast is None:
            forecast = load_from_file(self.options['forecast_file'])[0]
        forecast = to_utc(forecast)
        gen, cap = self.get_pvlive_data(forecast.index)
        gen.dropna(inplace=True)
        forecast = forecast.reindex(gen.index).dropna()
        cap = cap.reindex(gen.index).dropna()
        day_values = (gen.values.flatten()/cap.values.flatten()) > min_yield
        df = forecast[day_values]
        df.columns = ['forecast']
        df = df.assign(actual=gen[day_values])
        df = df.assign(cap=cap[day_values])
        return df

    @staticmethod
    def get_pvlive_data(forecast_index):
        """
            Extract pvlive data from api

            Parameters
            ----------
            forecast_index
                A Pandas index | DateTime | Horizon | from the forecast data
            Returns
            -------
            pv_data
        """
        datetimes = forecast_index.get_level_values(0).unique()
        horizons = forecast_index.get_level_values(1).unique()
        start = datetimes[0]
        end = datetimes[-1] + timedelta(hours=73)
        pvlive = PVLive()
        pvlive_data = pvlive.between(start, end, pes_id=0, extra_fields="installedcapacity_mwp")
        pvlive_df = pd.DataFrame(pvlive_data, columns=['region_id', 'datetime', 'gen', 'cap'])
        pvlive_df.index = pd.to_datetime(pvlive_df['datetime'])
        pvlive_df.drop(columns=['datetime', 'region_id'], inplace=True)
        pv_gen = shift_n_days(pvlive_df['gen'].values.reshape(-1, 1), horizons[0], horizons[-1]+1, reverse=True)
        pv_gen = pd.DataFrame(data=pv_gen,
                              index=pd.to_datetime(pvlive_df.index)).drop(columns=0).stack().reindex(forecast_index)
        pv_cap = shift_n_days(pvlive_df['cap'].values.reshape(-1, 1), horizons[0], horizons[-1] + 1, reverse=True)
        pv_cap = pd.DataFrame(data=pv_cap,
                              index=pd.to_datetime(pvlive_df.index)).drop(columns=0).stack().reindex(forecast_index)
        return pv_gen, pv_cap

    @staticmethod
    def r_squared(predictions, actuals):
        r"""
        Calculate the coefficient of determination (a.k.a R-Squared) [1]_.

        Parameters
        ----------
        `predictions` : numpy array of floats
            Predictions being tested.
        `actuals`: numpy array of floats
            Actual values corresponding to `predictions`. Must be same size as `predictions`.
        Returns
        -------
        float
            Coefficient of determination.
        Notes
        -----
        .. math::
            \begin{align*}
            y=Actuals,\quad f&=Predictions,\quad \bar{y}=\frac{1}{n}\sum_{i=1}^n{y_i}\\
            SS_{tot}&=\sum_i{(y_i-\bar{y})^2}\\
            SS_{res}&=\sum_i{(y_i-f_i)^2}\\
            R^2&=1-\frac{SS_{res}}{SS_{tot}}
            \end{align*}
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Coefficient_of_determination
        """
        mean_actual = np.mean(actuals, axis=1).values.reshape(-1, 1)
        ss_tot = np.nansum(np.power(actuals.values - mean_actual, 2), axis=1) + 0.0001
        ss_res = np.nansum(np.power(actuals.values - predictions.values, 2), axis=1)
        r_sqr = 1 - ss_res / ss_tot
        return pd.DataFrame(data=r_sqr, columns=['r2'], index=actuals.index)

    @staticmethod
    def wmape(predictions, actuals, norms=None, weights=None, axis=0):
        r"""
        Calculate the weighted Mean Absolute Percent Error (MAPE).

        Parameters
        ----------
        `predictions` : numpy array of floats
            Predictions being tested.
        `actuals` : numpy array of floats
            Actual values corresponding to `predictions`. Must be same size as `predictions`.
        `norms` : numpy array of floats
            Normalisation values. Must be same size as `predictions`. Default is to use `actuals`.
        `weights` : numpy array of floats
            Weighting values. Must be same size as `predictions`. Default is to use `actuals`.
        Returns
        -------
        float
            wMAPE.
        Notes
        -----
        .. math::
            \begin{gathered}
            y=Actuals,\quad f=Predictions,\quad n=Normalisations,\quad w=Weights\\
            \mathit{wMAPE}=
            \frac{\sum_i{w_i\times\mathrm{abs}\left(\frac{f_i-y_i}{n_i}\right)\times100\%}}{\sum_i{w_i}}
            \end{gathered}
        """
        norms = actuals if norms is None else norms
        weights = actuals if weights is None else weights
        mapes = np.abs((predictions - actuals) / norms) * 100.
        if axis == 1:
            wmape = np.sum((weights * mapes), axis=axis) / np.sum(weights, axis=axis)
        else:
            wmape = mapes
        return wmape

    def calc_heatmap(self, forecast, actuals, capacity):
        heatmap = dict()
        mapes = self.wmape(forecast, actuals, capacity, axis=0).to_frame()
        times = mapes.index.get_level_values(0) + pd.to_timedelta(30*mapes.index.get_level_values(1), 'm')
        mapes['month'] = times.month
        mapes['time'] = [t[:5] for t in times.time.astype(str)]
        mapes.set_index(['month', 'time'], inplace=True, drop=True)
        mapes_mean = mapes.groupby(['month', 'time']).mean().squeeze()
        heatmap_df = mapes_mean.unstack().fillna(0)
        heatmap['xlabels'] = heatmap_df.columns.values.tolist()
        heatmap['ylabels'] = heatmap_df.index.values.tolist()
        heatmap['values'] = heatmap_df.values.tolist()
        return heatmap

    @staticmethod
    def parse_options():
        """Parse command line options."""
        parser = argparse.ArgumentParser(description=("This is a command line interface (CLI) for "
                                                      "the PV-Forecast_Validation module"),
                                         epilog="Vlad Bondarenko 2020-03-13")
        parser.add_argument("--report-directory", dest="report_dir", action="store", type=str,
                            required=False, default="ValidationStatsReports",
                            help="Optionally specify the directory into which reports are printed "
                                 "(default is ./ValidationStatsReports/).")
        parser.add_argument("-f", dest="forecast_file", action="store", required=False,
                            help="Specify the pickl file from which to read forecast data")
        options = parser.parse_args()
        return options
