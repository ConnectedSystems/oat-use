Setup
=======

First create the project directory

```bash
mkdir sa-comparison
cd sa-comparison
```

Then clone the project:

```bash
git clone https://github.com/ConnectedSystems/sa-comparison.git
```

Assuming you are still in the project folder after cloning:

```bash
conda create -n sa-comparison python=3.6.6
pip install -r requirements.txt
```

Set up ipykernel with:

```bash
conda activate sa-comparison
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
