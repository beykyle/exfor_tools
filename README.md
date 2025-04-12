[![Python package](https://github.com/beykyle/exfor_tools/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/exfor_tools/actions/workflows/python-package.yml)
[![PyPI publisher](https://github.com/beykyle/exfor_tools/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/beykyle/exfor_tools/actions/workflows/pypi-publish.yml)

# exfor-tools
Some lightweight tools to grab data from the [EXFOR database](https://www-nds.iaea.org/exfor/) using the [x4i3 library](https://github.com/afedynitch/x4i3/), and organize it for visualization and use in model calibration and uncertainty quantification.

## quick start
```
 pip install exfor-tools
```

Package hosted at [pypi.org/project/exfor-tools/](https://pypi.org/project/exfor-tools/).

## testing

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

## examples and tutorials

Check out [examples/data_curation_tutorial.ipynb](https://github.com/beykyle/exfor_tools/blob/main/examples/dataset_curation_tutorial.ipynb)

This demonstrates how to query for and parse exfor entries, and curate and plot a data set, producing this figure: 

![](https://github.com/beykyle/exfor_tools/blob/main/assets/lead_208_pp_dxds.png)
