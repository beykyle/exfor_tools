[![Python package](https://github.com/beykyle/exfor_tools/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/exfor_tools/actions/workflows/python-package.yml)
[![PyPI publisher](https://github.com/beykyle/exfor_tools/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/beykyle/exfor_tools/actions/workflows/pypi-publish.yml)

# exfor-tools
Some lightweight tools to grab data from the [EXFOR database](https://www-nds.iaea.org/exfor/) using the [x4i3 library](https://github.com/afedynitch/x4i3/), and organize it for visualization and use in model calibration and uncertainty quantification.

## scope

Currently, `exfor_tools` supports most reactions in EXFOR, but only a small subset of the observables/quantities. Feel free to contribute! If it doesn't meet your needs check out the project it's built on, which is far more complete: [x4i3](https://github.com/afedynitch/x4i3/).

## quick start
```
 pip install exfor-tools
```

Package hosted at [pypi.org/project/exfor-tools/](https://pypi.org/project/exfor-tools/). Otherwise, for development, simply clone the repo and install locally:

```
git clone git@github.com:beykyle/exfor_tools.git --recurse-submodules
pip instal exfor_tools -e 
```

## tutorials

You can run the notebooks in the `examples/` directory to see how to use the package. To run the notebooks, some additional dependencies are required:

```
pip install -r examples/requirements.txt
```

The examples include:
-   [examples/introductory_tutorial.ipynb](https://github.com/beykyle/exfor_tools/blob/main/examples/introductory_tutorial.ipynb)
-   [examples/data_curation_tutorial.ipynb](https://github.com/beykyle/exfor_tools/blob/main/examples/dataset_curation_tutorial.ipynb)

These demonstrate how to query for and parse exfor entries, and curate and plot data sets. In the first one, you will produce this figure: 

![](https://github.com/beykyle/exfor_tools/blob/main/assets/lead_208_pp_dxds.png)

## test

The tests and the examples are one and the same. To run the tests, first install the dependencies for the notebooks:

```
pip install -r examples/requirements.txt
```

Then, to test that the notebooks run, use:

```
pytest --nbmake examples/
```

To test that they produce the expected results, use:

```
pytest --nbval-lax examples/
```

Note that there may be some difference in your installation, e.g. if you're using a different version of the EXFOR database, so the expected results may not be exactly the same as those in the tutorials. I will attempt to keep the notebooks in `examples/` up to date with new EXFOR releases.

By default, `x4i3` ships with the `2023-04-29` EXFOR release. There are a set of notebooks with stored outputs valid for that release in `examples/examples_2023_release/`. These are used in the github actions. If you haven't updated to a more recent release but you would like to run the test, then simply run:


```
pytest --nbval-lax examples/examples_2023_release/
```


## updating the EXFOR data base

First, download your desired version `<exfor-YYYY.zip>` from here: [https://nds.iaea.org/nrdc/exfor-master/list.html](https://nds.iaea.org/nrdc/exfor-master/list.html). The latest is recomended. Then:

```sh
bash update_database.sh </path/to/exfor-XXXX.zip> --db-dir </path/where/db/should/go/>
```

This will extract and process the data to `</path/where/db/should/go/unpack_exfor-YYYY/X4-YYYY-12-31>`, setting the environment variable `$X43I_DATAPATH` accordingly. `x4i3` uses this environment variable to find the database on `import`, so you should add this to your environment setup. If you use bash, this will look something like this:

```sh
echo export X43I_DATAPATH=$X43I_DATAPATH >> ~/.bashrc
```

This functionality for modifying the database used by `x4i3` is provided in [x4i3_tools](https://github.com/afedynitch/x4i3_tools), which is included as a submodule.
