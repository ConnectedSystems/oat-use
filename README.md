Setup
=======

An environment file for Windows is provided. If installing for other OS platforms, try removing windows specific packages.

```bash
conda env create --file win_environment.yml -n sa-comparison
```

Install SALib with:

```bash
pip install salib==1.2
```

Set up ipykernel with:

```bash
python -m ipykernel install --name sa-comparison --display-name "Python (sa-comparison)"
```

Notebooks
=========

Notebooks for the study are available in the `notebooks` directory.

```bash
conda activate sa-comparison
cd notebooks
jupyter notebook
```
