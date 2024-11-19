## Training

To train the model open terminal in `api` directory and write command `python training_v1.py`.

It seeks for `data_v1.xlsx` file in `api/data` directory.

It saves model in `api/models`

## Starting the server

To start the server `waitress` needs to be installed.

To start server open `start_server.bat` batch file.

Server will be hosted on `http://127.0.0.1:5000`.

## Dependencies

Required python packages:
- openpyxl;
- pandas;
- numpy;
- scikit-learn 1.5.2;
- dill;
- Flask;
- waitress.
