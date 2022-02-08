# AudioClean Backend

Source code of the AudioClean backend

## Important notes

* Runs in Python 3.8 because PyTorch doesn't have a build for 3.9 and somethimes raises errors when the package is installing.

* The backend runs by default in the localhost using the port 8000, this is to ensure a correct communication between the frontend and the backend.

* In order to correctly create a segment this must have a duration greater than 25 ms.

* The backend uses a model called "orca-clean.pk" that needs to be stored in the path ..\Backend AudioClean\controller\ORCA_CLEAN, this model can be requested to Christian Bergler author of ORCA-CLEAN: A Deep Denoising Toolkit for Killer Whale Communication (https://www.isca-speech.org/archive/Interspeech_2020/abstracts/1316.html)

```
@inproceedings{Bergler-OC-2020,
  author={Christian Bergler and Manuel Schmitt and Andreas Maier and Simeon Smeele and Volker Barth and Elmar Nöth},
  title={ORCA-CLEAN: A Deep Denoising Toolkit for Killer Whale Communication},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={1136--1140},
  doi={10.21437/Interspeech.2020-1316},
  url={http://dx.doi.org/10.21437/Interspeech.2020-1316}
}
```

## Getting Started

### Installing

Create a new enviroment is the best to avoid issues in the installation of the packages
```
conda create --no-default-packages -n NAME_OF_THE_ENV python
```
* After the creation of the enviroment, the packages needs to be installed, then run:
```
pip install -r requirements.txt
```

### Executing program

In the folder ..\Backend AudioClean\controller run the server using the file [manage.py](https://github.com/nestorcalvo/Backend-AudioClean/blob/master/controller/manage.py),
by default the server will run in the port 8000
```
python manage.py makemigrations && python manage.py migrate && python manage.py runserver 
```

Once the backend is running, start the frontend.
