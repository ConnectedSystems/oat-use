Setup
=======

First create the project directory

```bash
$ mkdir sa-comparison
$ cd sa-comparison
```

Then clone the project:

```bash
$ git clone https://github.com/ConnectedSystems/sa-comparison.git
```

Assuming you are still in the project folder after cloning:

```bash
$ conda create -n sa-comparison python=3.6.6
$ conda activate sa-comparison
$ pip install -r requirements.txt
```

Set up ipykernel with:

```bash
$ python -m ipykernel install --name sa-comparison --display-name "Python (sa-comparison)"
```

Notebooks
=========

Notebooks are found in the `notebooks` directory.

Once the `conda` environment is activated:

```bash
$ cd notebooks
$ jupyter notebook
```
