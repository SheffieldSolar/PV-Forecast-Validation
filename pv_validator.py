#!/usr/bin/env python3
"""
A Flask UI to validate a PV forecast against PV_Live.

- Jamie Taylor <jamie.taylor@sheffield.ac.uk>
- Vlad Bondarenko <vbondarenko1@sheffield.ac.uk>
- First Authored: 2020-03-13
"""

import os
import pickle
from io import StringIO
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
    fbase_selected = request.args.get("fbase", "07:00,10:00").split(",")
    hm_min = request.args.get("hm_min", 0.)
    hm_max = request.args.get("hm_max", 10.)
    cache_dir = os.path.join(ROOT_PATH, "cache")
    if cache_id is None:
        now_ = datetime.utcnow()
        now = now_.strftime("%Y-%m-%d %H:%M:%S UTC")
        cache_id = now_.strftime("%Y%m%d%H%M%S")
        stats_cache_file = os.path.join(cache_dir, "stats_{}.p".format(cache_id))
        pvf_data, forecast_bases, entity_type, entity_ids = read_forecast(request, cache_dir, cache_id)
        start = forecast_bases.min()
        end = forecast_bases.max()
        validation = Validation()
        data = validation.run_validation(entity_type, entity_ids, forecast=pvf_data)
        fbase_available = []
        for entity_id in data["data"]:
            for type_ in data["data"][entity_id]:
                fbase_available += data["data"][entity_id][type_].keys()
                for fbase in data["data"][entity_id][type_]:
                    this = data["data"][entity_id][type_][fbase]
                    data["data"][entity_id][type_][fbase]["predicted_vs_actual"] = [
                        [x, y] for x, y in zip(this["actual"], this["predicted"])
                    ]
                    linear_fit = np.poly1d(np.polyfit(this["actual"], this["predicted"], 1))
                    minim = min(this["actual"])
                    maxim = max(this["actual"])
                    data["data"][entity_id][type_][fbase]["linear_fit"] = [[minim, linear_fit(minim)],
                                                                           [maxim, linear_fit(maxim)]]
                    data["data"][entity_id][type_][fbase]["heatmap"]["heatmap_xyz"] = []
                    for j, row in enumerate(this["heatmap"]["values"]):
                        for i, val in enumerate(row):
                            data["data"][entity_id][type_][fbase]["heatmap"]["heatmap_xyz"].append(
                                [i, j, val]
                            )
        fbase_available = list(set(fbase_available))
        fbase_available.sort()
        input_filename = request.files["dataFile"].filename
        with open(stats_cache_file, "wb") as fid:
            pickle.dump((now, start, end, fbase_available, data, input_filename, entity_type, entity_ids), fid)
    else:
        stats_cache_file = os.path.join(cache_dir, "stats_{}.p".format(cache_id))
        with open(stats_cache_file, "rb") as fid:
            now, start, end, fbase_available, data, input_filename, entity_type, entity_ids = pickle.load(fid)
    data_ = copy.deepcopy(data["data"])
    for entity_id in data["data"]:
        for type_ in data["data"][entity_id]:
            for fb in data["data"][entity_id][type_]:
                if fb not in fbase_selected:
                    data_[entity_id][type_].pop(fb)
    return render_template("validation_report.html", data=data_, report_timestamp=now, start=start,
                           end=end, fbase_selected=fbase_selected, entity_type=entity_type,
                           entity_ids=entity_ids, cache_id=cache_id,
                           fbase_available=fbase_available, hm_min=hm_min, hm_max=hm_max,
                           input_filename=input_filename, now=TIME.time())


def read_forecast(request, cache_dir, cache_id):
    data_cache_file = os.path.join(cache_dir, f"data_{cache_id}.p")
    if request.method == "POST" and "dataFile" in request.files:
        # Expected columns: forecast_base,horizon,entity_id,generation
        entity_type = request.form["entityType"]
        dummy = StringIO(request.files["dataFile"].stream.read().decode())
        pvf_data = pd.read_csv(dummy)
        pvf_data.columns = pvf_data.columns.str.lower()
        if "entity_id" not in pvf_data.columns:
            pvf_data["entity_id"] = 0
        pvf_data.dropna(inplace=True)
        # Enforce correct types:
        pvf_data["forecast_base"] = pd.to_datetime(pvf_data["forecast_base"], utc=True)
        pvf_data["horizon"] = pvf_data["horizon"].astype("int64")
        pvf_data["entity_id"] = pvf_data["entity_id"].astype("int64")
        pvf_data["generation"] = pvf_data["generation"].astype("float64")
        pvf_data.set_index(["entity_id", "forecast_base", "horizon"], inplace=True)
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        with open(data_cache_file, "wb") as fid:
            pickle.dump((entity_type, pvf_data), fid)
    else:
        with open(data_cache_file, "rb") as fid:
            entity_type, pvf_data = pickle.load(fid)
    entity_ids = pvf_data.index.get_level_values(0).unique()
    forecast_bases = pvf_data.index.get_level_values(1)
    return pvf_data, forecast_bases, entity_type, entity_ids
