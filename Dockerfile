FROM python:3.7
WORKDIR /pv_forecast_validation

COPY requirements.txt /pv_forecast_validation/requirements.txt

RUN pip install --no-cache-dir -r /pv_forecast_validation/requirements.txt > /dev/null

COPY . /pv_forecast_validation/

CMD ["/bin/bash", "/pv_forecast_validation/launch_ui.sh"]
