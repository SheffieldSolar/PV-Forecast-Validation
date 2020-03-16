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
from flask import Flask, request, url_for, redirect
from flask.templating import render_template
import pandas as pd
import numpy as np


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_PATH, "uploads")

APP = Flask(__name__)
APP.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@APP.route("/", methods=["GET", "POST"])
def home_page():
    return render_template("home_page.html")

@APP.route("/validate_pv_forecast", methods=["GET", "POST"])
def validate_pv_forecast():
    cache_dir = os.path.join(ROOT_PATH, "cache")
    cache_file = os.path.join(cache_dir, "data.p")
    if request.method == "POST" and "dataFile" in request.files:
        pvf_data = pd.read_csv(request.files["dataFile"].stream, names=["fbase", "horizon",
                                                                        "generation_mw"])
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        with open(cache_file, "wb") as fid:
            pickle.dump(pvf_data, fid)
    else:
        with open(cache_file, "rb") as fid:
            pvf_data = pickle.load(fid)
    # data = vlad_code(pvf_data)
    ##### USE TEST DATA UNTIL VLAD's CODE IS IMPLEMENTED #####
    import json
    with open("data.json", "r") as fid:
        data = json.load(fid)
    ##########################################################
    for type in data["data"]["Region0"]:
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
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return render_template("validation_report.html", data=data["data"]["Region0"], report_timestamp=now)
