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
    hm_max = request.args.get("hm_max", 15.)
    cache_dir = os.path.join(ROOT_PATH, "cache")
    data_cache_file = os.path.join(cache_dir, "data.p")
    if cache_id is None:
        now_ = datetime.utcnow()
        now = now_.strftime("%Y-%m-%d %H:%M:%S UTC")
        cache_id = now_.strftime("%Y%m%d%H%M%S")
        stats_cache_file = os.path.join(cache_dir, "stats_{}.p".format(cache_id))
        if request.method == "POST" and "dataFile" in request.files:
            pvf_data = pd.read_csv(request.files["dataFile"].stream,
                                   names=["fbase", "horizon", "generation_mw"],
                                   index_col=['fbase', 'horizon'], parse_dates=['fbase'])
            if not os.path.isdir(cache_dir):
                os.mkdir(cache_dir)
            with open(data_cache_file, "wb") as fid:
                pickle.dump(pvf_data, fid)
        else:
            with open(data_cache_file, "rb") as fid:
                pvf_data = pickle.load(fid)
        dates = pvf_data.index.get_level_values(0)
        start = dates.min()
        end = dates.max()
        validation = Validation(options={'forecast_file': 'results/forecast.p'})
        data = validation.run_validation(forecast=pvf_data)
        fbase_available = []
        for type in data["data"]["Region0"]:
            fbase_available += data["data"]["Region0"][type].keys()
            for fbase in data["data"]["Region0"][type]:
                this = data["data"]["Region0"][type][fbase]
                data["data"]["Region0"][type][fbase]["predicted_vs_actual"] = [
                    [x, y] for x, y in zip(this["actual"], this["predicted"])
                ]
                linear_fit = np.poly1d(np.polyfit(this["actual"], this["predicted"], 1))
                minim = min(this["actual"])
                maxim = max(this["actual"])
                data["data"]["Region0"][type][fbase]["linear_fit"] = [[minim, linear_fit(minim)],
                                                                      [maxim, linear_fit(maxim)]]
                data["data"]["Region0"][type][fbase]["heatmap"]["heatmap_xyz"] = []
                for j, row in enumerate(this["heatmap"]["values"]):
                    for i, val in enumerate(row):
                        data["data"]["Region0"][type][fbase]["heatmap"]["heatmap_xyz"].append(
                            [i, j, val]
                        )
        fbase_available = list(set(fbase_available))
        fbase_available.sort()
        with open(stats_cache_file, "wb") as fid:
            pickle.dump((now, start, end, fbase_available, data), fid)
    else:
        stats_cache_file = os.path.join(cache_dir, "stats_{}.p".format(cache_id))
        with open(stats_cache_file, "rb") as fid:
            now, start, end, fbase_available, data = pickle.load(fid)
    data_ = copy.deepcopy(data["data"]["Region0"])
    for type in data["data"]["Region0"]:
        for fb in data["data"]["Region0"][type]:
            # print("{}    {}    Mean r-squared: {}    Median r-squared: {}"
                  # .format(type, fb, np.mean([x for x in data_[type][fb]["r_squared"] if x >= 0]),
                          # np.median([x for x in data_[type][fb]["r_squared"] if x >= 0])))
            if fb not in fbase_selected:
                data_[type].pop(fb)
    return render_template("validation_report.html", data=data_,
                           report_timestamp=now, start=start, end=end, fbase_selected=fbase_selected,
                           cache_id=cache_id, fbase_available=fbase_available, hm_max=hm_max)
