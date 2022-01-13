# ML-RT version 2
Radiative transfer with advanced machine learning methods


## Setup

### Software requirements

We assume your system is equipped with the following dependencies:

* Python 3.8 or newer
* bash
* wget
* unzip
* md5sum (optional)

#### System packages
On Debian or Debian-derivatives, e.g. Ubuntu, the required packages should be part of the base installation 
but can be installed using the default package manager if necessary with the following command:
```bash
sudo apt install wget unzip md5sum
```
#### Python modules
Furthermore, the following Python packages are needed:

* pytorch
* numpy (1.20.x for now)
* numba (used to generate SEDs)
* pydoe
* matplotlib
* deepxde
* tensorboard


##### pip
The Python dependencies can be installed with `pip` like so:
```bash
pip3 install -r requirements.txt
```

##### conda
In Anaconda (or Miniconda) environments the requirements can be installed like so:
```bash
conda config --add channels conda-forge
conda install --yes --file requirements_conda.txt
```