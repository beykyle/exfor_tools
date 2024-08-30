[![Python package](https://github.com/beykyle/exfor_tools/actions/workflows/python-package.yml/badge.svg)](https://github.com/beykyle/exfor_tools/actions/workflows/python-package.yml)

# exfor-tools
Some lightweight tools to grab data from the [EXFOR database](https://www-nds.iaea.org/exfor/) using the [x4i3 library](https://github.com/afedynitch/x4i3/), and organize it for visualization and use in the calibration of optical potentials

## quick start
```
 pip install exfor-tools
```

Package hosted at [pypi.org/project/exfor-tools/](https://pypi.org/project/exfor-tools/).

## testing

TODO


## examples and tutorials

Check out [examples/](https://github.com/beykyle/exfor_tools/tree/main/examples)



```python
all_entries_lead208_pp = get_exfor_differential_data(
    target=(208, 82),
    projectile=(1, 1),
    quantity="dXS/dA",
    product="EL",
    energy_range=[10, 60], # MeV
)
print(f"Found {len(all_entries_lead208_pp.keys())} entries")
print(all_entries_lead208_pp.keys())
```

should print
```
Found 14 entries
dict_keys(['C0893', 'C1019', 'C2700', 'E1846', 'O0142', 'O0157', 'O0166', 'O0187', 'O0191', 'O0208', 'O0225', 'O0287', 'O0391', 'O0598'])
```

Now we can plot them. 

```
measurements_condensed = sort_measurements_by_energy(all_entries_lead208_pp)
fig, ax = plt.subplots(1, 1, figsize=(6, 12))
entry = all_entries_lead208_pp["C0893"]
entry.plot_experiment(
    ax,
    offsets=50,
    measurements=measurements_condensed,
    label_offset_factor=2,
    label_hloc_deg=150,
    label_energy_err=False,
    label_offset=False,
)
```


This should produce the following figure: 

![](https://github.com/beykyle/exfor_tools/blob/main/assets/lead_208_pp_dxds.png)



