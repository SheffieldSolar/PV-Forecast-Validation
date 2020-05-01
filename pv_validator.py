#!/usr/bin/env python3
"""
A Flask UI to validate a PV forecast against PV_Live.

- Jamie Taylor <jamie.taylor@sheffield.ac.uk>
- Vlad Bondarenko <vbondarenko1@sheffield.ac.uk>
- First Authored: 2020-03-13
"""

import os
import pickle
from datetime import datetime
import time as TIME
import copy
from flask import Flask, request, url_for, redirect
from flask.templating import render_template
import pandas as pd
import numpy as np

from validation import Validation

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_PATH, "uploads")

APP = Flask(__name__)
APP.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@APP.route("/", methods=["GET", "POST"])
def home_page():
    return render_template("home_page.html")


@APP.route("/validate_pv_forecast", methods=["GET", "POST"])
def validate_pv_forecast():
    cache_id = request.args.get("cache_id", None)
    fbase_selected = request.args.get("fbase", "07:00").split(",")
    region_selected = request.args.get("region", "0").split(",")
    hm_min = request.args.get("hm_min", 0.)
    hm_max = request.args.get("hm_max", 15.)
    cache_dir = os.path.join(ROOT_PATH, "cache")
    if cache_id is None:
        now_ = datetime.utcnow()
        now = now_.strftime("%Y-%m-%d %H:%M:%S UTC")
        cache_id = now_.strftime("%Y%m%d%H%M%S")
        stats_cache_file = os.path.join(cache_dir, "stats_{}.p".format(cache_id))
        pvf_data, forecast_bases, regions = read_forecast(request, cache_dir, cache_id)
        start = forecast_bases.min()
        end = forecast_bases.max()
        validation = Validation()
        data = validation.run_validation(regions, forecast=pvf_data)
        fbase_available = []
        for region in data["data"]:
            for type_ in data["data"][region]:
                fbase_available += data["data"][region][type_].keys()
                for fbase in data["data"][region][type_]:
                    this = data["data"][region][type_][fbase]
                    data["data"][region][type_][fbase]["predicted_vs_actual"] = [
                        [x, y] for x, y in zip(this["actual"], this["predicted"])
                    ]
                    linear_fit = np.poly1d(np.polyfit(this["actual"], this["predicted"], 1))
                    minim = min(this["actual"])
                    maxim = max(this["actual"])
                    data["data"][region][type_][fbase]["linear_fit"] = [[minim, linear_fit(minim)],
                                                                           [maxim, linear_fit(maxim)]]
                    data["data"][region][type_][fbase]["heatmap"]["heatmap_xyz"] = []
                    for j, row in enumerate(this["heatmap"]["values"]):
                        for i, val in enumerate(row):
                            data["data"][region][type_][fbase]["heatmap"]["heatmap_xyz"].append(
                                [i, j, val]
                            )
        fbase_available = list(set(fbase_available))
        fbase_available.sort()
        input_filename = request.files["dataFile"].filename
        with open(stats_cache_file, "wb") as fid:
            pickle.dump((now, start, end, fbase_available, data, input_filename), fid)
    else:
        stats_cache_file = os.path.join(cache_dir, "stats_{}.p".format(cache_id))
        with open(stats_cache_file, "rb") as fid:
            now, start, end, fbase_available, data, input_filename = pickle.load(fid)
    data_ = copy.deepcopy(data["data"])
    for region in data["data"]:
        # if region not in region_selected:
            # data_.pop(region)
            # continue
        for type_ in data["data"][region]:
            for fb in data["data"][region][type_]:
                if fb not in fbase_selected:
                    data_[region][type_].pop(fb)
    return render_template("validation_report.html", data=data_, report_timestamp=now, start=start,
                           end=end, fbase_selected=fbase_selected, region_selected=region_selected,
                           cache_id=cache_id, fbase_available=fbase_available, hm_min=hm_min,
                           hm_max=hm_max, input_filename=input_filename)

def read_forecast(request, cache_dir, cache_id):
    data_cache_file = os.path.join(cache_dir, f"data_{cache_id}.p")
    if request.method == "POST" and "dataFile" in request.files:
        # Expected columns: forecast_base,horizon,region,generation
        pvf_data = pd.read_csv(request.files["dataFile"].stream)
        pvf_data.columns = pvf_data.columns.str.lower()
        if "region" not in pvf_data.columns:
            pvf_data["region"] = 0
        pvf_data.dropna(inplace=True)
        # Enforce correct types:
        pvf_data["forecast_base"] = pd.to_datetime(pvf_data["forecast_base"], utc=True)
        pvf_data["horizon"] = pvf_data["horizon"].astype("int64")
        pvf_data["region"] = pvf_data["region"].astype("int64")
        pvf_data["generation"] = pvf_data["generation"].astype("float64")
        pvf_data.set_index(["region", "forecast_base", "horizon"], inplace=True)
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        with open(data_cache_file, "wb") as fid:
            pickle.dump(pvf_data, fid)
    else:
        with open(data_cache_file, "rb") as fid:
            pvf_data = pickle.load(fid)
    regions = pvf_data.index.get_level_values(0).unique()
    forecast_bases = pvf_data.index.get_level_values(1)
    return pvf_data, forecast_bases, regions
