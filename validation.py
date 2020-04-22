#!/usr/bin/env python3
"""
Validate a PV forecast against PV_Live.

- Vlad Bondarenko <vbondarenko1@sheffield.ac.uk>
- Jamie Taylor <jamie.taylor@sheffield.ac.uk>
- First Authored: 2020-03-13
"""

from datetime import timedelta, time
import pandas as pd
import numpy as np

from pvlive_api import PVLive
from helper import shift_n_days


class Validation:

    def run_validation(self, regions, forecast, min_yield=0.05, val='full'):
        """
            Runs full validation for all regions

            Parameters
            ----------
            regions: list
                A list of region pes ids for which forecast data will be provided

            forecast: pd.DataFrame, optional
                Forecast data to be validated with index | Region | Datetime | Horizon |
            
            min_yield: float
                Cut out all the values with yield below min_yield
            
            val: {'full', 'fast'}
                Perform either full validation or fast returning only one value

            Returns
            -------
            val_data: dict
                Dictionary with all of the validation results ready for plotting
        """
    
        val_data = {'data': {}}
        display_data = val_data['data']
        
        for region in regions:
            
            region_str = f'Region{region}'
            display_data[region_str] = dict()
            val_data[region_str] = dict()
            region_forecast = forecast.loc[region]
            data = self.get_data(region_forecast, region, min_yield)
            val_data[region_str]['wmape'] = self.wmape(data['forecast'], data['actual'], data['cap']).mean()
            val_data[region_str]['r_squared'] = self.r_squared(data['forecast'], data['actual'])
            
            if val == 'full':
                
                display_data[region_str]['intraday'] = dict()
                display_data[region_str]['day1'] = dict()

                horizons = data.index.get_level_values(1)
                dt = data.index.get_level_values(0)
                
                for base in dt.hour.unique():
                    
                    base_str = str(time(hour=base))[:5]
            
                    intraday = data[(dt.hour == base) & (horizons.isin(np.arange(0, 48 - base)))]
                    dayp1 = data[(dt.hour == base) & (horizons.isin(np.arange(48 - (base * 2), 96 - (base * 2))))]
            
                    period_names = ['intraday', 'day1']
                    for i, period_data in enumerate([intraday, dayp1]):
            
                        display_data[region_str][period_names[i]][base_str] = dict()
                        display_period = display_data[region_str][period_names[i]][base_str]
            
                        pred, actual, cap = period_data['forecast'], period_data['actual'], period_data['cap']
                        pred_u, actual_u, cap_u = pred.unstack(), actual.unstack(), cap.unstack()
            
                        errors = pd.DataFrame(index=pred_u.index)
                        errors['mape'] = self.wmape(pred_u, actual_u, cap_u, axis=1)
                        errors['r_squared'] = self.r_squared(pred, actual)
            
                        display_period['r_squared'] = errors['r_squared']
                        display_period['mape'] = errors['mape'].values.tolist()
                        display_period['actual'] = actual.values.tolist()
                        display_period['predicted'] = pred.values.tolist()
                        display_period['heatmap'] = self.calc_heatmap(pred, actual, cap)

        return val_data

    def get_data(self, forecast, region, min_yield):
        """
            Get pvlive data, fixes indexes to be same and keeps day values

            Parameters
            ----------
            forecast: pd.DataFrame
                A Pandas DataFrame with index | DateTime | Horizon |

            region: int
                Pes region for which the forecast has been produces
            
            min_yield: float
                Cut out all the values with yield below provided

            Returns
            -------
            df: pd.DataFrame
                DataFrame with actual and forecast data
        """

        gen, cap = self.get_pvlive_data(forecast.index, region)
        gen.dropna(inplace=True)
        forecast = forecast.reindex(gen.index).dropna()
        cap = cap.reindex(gen.index).dropna()

        day_values = (gen.values.flatten()/cap.values.flatten()) > min_yield
        df = forecast[day_values].copy()
        if not isinstance(df, pd.DataFrame):
            df = df.to_frame()
        df.columns = ['forecast']
        df['actual'] = gen.loc[day_values]
        df['cap'] = cap[day_values]

        return df

    @staticmethod
    def get_pvlive_data(forecast_index, region):
        """
            Extract pvlive data from api

            Parameters
            ----------
            forecast_index: pd.MultiIndex
                A Pandas index | DateTime | Horizon | from the forecast data
            
            region: int
                Pes region to get the pv data from
                
            Returns
            -------
            pv_data
        """

        datetimes = forecast_index.get_level_values(0).unique()
        horizons = forecast_index.get_level_values(1).unique()

        start = datetimes[0]
        end = datetimes[-1] + timedelta(hours=73)
        pvlive = PVLive()

        pvlive_data = pvlive.between(start, end, pes_id=region, extra_fields="installedcapacity_mwp")
        pvlive_df = pd.DataFrame(pvlive_data, columns=['region_id', 'datetime', 'gen', 'cap'])
        pvlive_df.index = pd.to_datetime(pvlive_df['datetime'])
        pvlive_df.drop(columns=['datetime', 'region_id'], inplace=True)

        pv_gen = shift_n_days(pvlive_df['gen'].values.reshape(-1, 1), horizons[0], horizons[-1]+1, reverse=True)
        pv_gen = pd.DataFrame(data=pv_gen, index=pd.to_datetime(pvlive_df.index))\
            .drop(columns=0).stack().reindex(forecast_index)

        pv_cap = shift_n_days(pvlive_df['cap'].values.reshape(-1, 1), horizons[0], horizons[-1] + 1, reverse=True)
        pv_cap = pd.DataFrame(data=pv_cap, index=pd.to_datetime(pvlive_df.index))\
            .drop(columns=0).stack().reindex(forecast_index)
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
        mean_actual = np.mean(actuals)
        ss_tot = np.sum(np.power(actuals - mean_actual, 2))
        ss_res = np.sum(np.power(actuals - predictions, 2))
        r_sqr = 1 - ss_res / ss_tot
        return r_sqr

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
