$env:FLASK_APP = "$PSScriptRoot\pv_validator.py"
$env:FLASK_ENV = "development"
#$env:FLASK_APP
start  http://127.0.0.1:5000/
python -m flask run