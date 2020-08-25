Setup
=======

Clone the necessary repositories:

```bash
$ git clone https://github.com/ConnectedSystems/sa-comparison.git
$ git clone https://github.com/ConnectedSystems/SALib.git --branch radial-oat-method --single-branch salib-roat
```

Set up environment from the project folder:

```bash
$ cd oat-use
$ conda create -n oat-use python=3.6.6 -y
$ conda activate oat-use
$ pip install -r requirements.txt
$ cd ..
```

Install the specific branch of SALib used for the study:

```bash
$ cd salib-roat
$ pip install -e .
$ cd ..
```

Set up ipykernel with:

```bash
$ python -m ipykernel install --name oat-use --display-name "Python (oat-use)"
```

Notebooks
=========

Notebooks are found in the `notebooks` directory and are labelled order.

Once the `conda` environment is activated:

```bash
$ cd notebooks
$ jupyter notebook
```
