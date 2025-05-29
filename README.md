[![Python package](https://github.com/beykyle/exfor_tools/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/exfor_tools/actions/workflows/python-package.yml)
[![PyPI publisher](https://github.com/beykyle/exfor_tools/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/beykyle/exfor_tools/actions/workflows/pypi-publish.yml)

# exfor-tools
Some lightweight tools to grab data from the [EXFOR database](https://www-nds.iaea.org/exfor/) using the [x4i3 library](https://github.com/afedynitch/x4i3/), and organize it for visualization and use in model calibration and uncertainty quantification.

## use case

You have a reaction model $f(x,\alpha)$ and you would like to find some data $y(x)$ to constrain the $\alpha$. Here $x$ can be energy, angle, type of reaction, projectile and target, etc.  You would like to do this in a statistically rigorous way, in which you take into account various types of uncertainties, including systematic uncertainties that introduce correlations between $y(x_i)$ and $y(x_j)$. You would also like to do this with large data sets comprised of many different experiments. In other words, you would like to curate a data set $y$ -- composed of reaction observables like differential cross sections -- and have the information required to construct a covariance matrix for it. And you would like it to be sorted into computationally convenient data structures that you can use for visualization, or comparison to your model. You've come to the right place.



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

## examples and tutorials

Check out the tutorials:
-   [examples/introductory_tutorial.ipynb](https://github.com/beykyle/exfor_tools/blob/main/examples/introductory_tutorial.ipynb)
-   [examples/data_curation_tutorial.ipynb](https://github.com/beykyle/exfor_tools/blob/main/examples/dataset_curation_tutorial.ipynb)

These demonstrate how to query for and parse exfor entries, and curate and plot data sets. In the first one, you will produce this figure: 

![](https://github.com/beykyle/exfor_tools/blob/main/assets/lead_208_pp_dxds.png)

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

