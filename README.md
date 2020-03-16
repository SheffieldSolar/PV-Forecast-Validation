# PV-Forecast-Validation

Python code to produce validation statistics for a national or regional GB PV forecast.

## How does it work?

The code is designed to be run inside a [Docker container](https://hub.docker.com/repository/docker/sheffieldsolar/pv_forecast_validation) and will launch a local Flask development server on your machine. You can then visit http://127.0.0.1:5000 nad use the web niterface to upload some historical PV forecast data. The code will fetch the corresponding "actual" data from the [PV_Live API](https://www.solar.sheffield.ac.uk/pvlive/) and calculate various error metrics, before presenting them as interactive graphs in your browser. The graphical reports can be exported to PDF using your browser built-in "print-to-PDF" function.

## How do I get set up?

* Install Docker on your machine: https://docs.docker.com/install/
* Clone this repository to your machine.

## How do I get started?

In the terminal/command-prompt, change directory (`cd <path-to-directory>`) into the location of this Git repo, then run the following command:

`docker run -it --rm -p 5000:5000 sheffieldsolar/pv_forecast_validation:latest`

Visit http://127.0.0.1:5000 in your browser - that's it!
