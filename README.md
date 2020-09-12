A use for One-At-a-Time sensitivity analysis: Diagnostic testing of integrated models

Companion code for the paper detailing the analysis approach and figure generation.


Setup
=======

Clone the necessary repositories:

```bash
$ git clone https://github.com/ConnectedSystems/oat-use.git
$ git clone https://github.com/ConnectedSystems/SALib.git --branch radial-oat-method --single-branch salib-roat
$ cd salib-roat
$ git checkout da99fdaed05c29e98ab8d7685d2c6ad3783ce946
$ cd ..
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

The full instructions as above:

```bash
$ git clone https://github.com/ConnectedSystems/oat-use.git
$ git clone https://github.com/ConnectedSystems/SALib.git --branch radial-oat-method --single-branch salib-roat 
$ cd salib-roat
$ git checkout da99fdaed05c29e98ab8d7685d2c6ad3783ce946
$ cd ..

$ conda create -n oat-use python=3.6.6 -y
$ conda activate oat-use
$ cd oat-use
$ pip install -r requirements.txt
$ cd ..
$ cd salib-roat
$ pip install -e .
$ cd ..
$ python -m ipykernel install --name oat-use --display-name "Python (oat-use)"
```

Notebooks
=========

Notebooks are found in the `notebooks` directory and are labelled in order of use.

Miscellaneous notebooks are unnumbered and are included for posterity.

To view notebooks locally, it is assumed jupyter `notebook` or `lab` is installed.

```bash
pip install jupyterlab
```

Once the `conda` environment is activated:

```bash
$ cd notebooks
$ jupyter lab
```

Scripts
=========

Scripts used to generate samples are found in the `scripts` directory.
These are to retain a record of sample provenance and do not need to be run.
