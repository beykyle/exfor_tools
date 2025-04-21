from functools import reduce

import numpy as np

from x4i3.exfor_column_parsing import (
    X4ColumnParser,
    X4IndependentColumnPair,
    angDistUnits,
    angleParserList,
    baseDataKeys,
    condenseColumn,
    dataTotalErrorKeys,
    energyUnits,
    errorSuffix,
    frameSuffix,
    incidentEnergyParserList,
    noUnits,
    percentUnits,
    resolutionFWSuffix,
    resolutionHWSuffix,
    variableSuffix,
    X4MissingErrorColumnPair,
)


# these are the supported quantities at the moment
quantity_matches = {
    "dXS/dA": [["DA"], ["PAR", "DA"]],
    "dXS/dRuth": [["DA", "RTH"], ["DA", "RTH/REL"]],
    "Ay": [["POL/DA", "ANA"]],
}
quantities = list(quantity_matches.keys())

quantity_symbols = {
    ("DA",): r"$\frac{d\sigma}{d\Omega}$",
    ("DA", "RTH"): r"$\sigma / \sigma_{Rutherford}$",
    ("DA", "RTH/REL"): r"$\sigma / \sigma_{Rutherford}$",
    ("POL/DA", "ANA"): r"$A_y$",
}

unit_symbols = {"no-dim": "unitless", "barns/ster": "b/Sr"}
energyExParserList = [
    X4MissingErrorColumnPair(
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in variableSuffix]
            + ["E-EXC" + s for s in variableSuffix],
            match_units=energyUnits,
        ),
        None,
    ),
    X4IndependentColumnPair(
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in variableSuffix]
            + ["E-EXC" + s for s in variableSuffix],
            match_units=energyUnits,
        ),
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in errorSuffix]
            + ["E-EXC" + s for s in errorSuffix],
            match_units=energyUnits + percentUnits,
        ),
    ),
    X4IndependentColumnPair(
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in variableSuffix]
            + ["E-EXC" + s for s in variableSuffix],
            match_units=energyUnits,
        ),
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in resolutionFWSuffix]
            + ["E-EXC" + s for s in resolutionFWSuffix],
            match_units=energyUnits + percentUnits,
            scale_factor=0.5,
        ),
    ),
    X4IndependentColumnPair(
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in variableSuffix]
            + ["E-EXC" + s for s in variableSuffix],
            match_units=energyUnits,
        ),
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in resolutionHWSuffix]
            + ["E-EXC" + s for s in resolutionHWSuffix],
            match_units=energyUnits + percentUnits,
        ),
    ),
]


def parse_differential_data(
    data_set,
    data_error_columns=["DATA-ERR"],
):
    r"""
    Extract differential cross section (potentially as ratio to Rutherford)
    """
    data_parser = X4ColumnParser(
        match_labels=reduce(
            lambda x, y: x + y,
            [[b + s for s in variableSuffix + frameSuffix] for b in baseDataKeys],
        ),
        match_units=angDistUnits + noUnits,
    )
    match_idxs = data_parser.allMatches(data_set)
    if len(match_idxs) != 1:
        raise ValueError(f"Expected only one DATA column, found {len(match_idxs)}")
    iy = match_idxs[0]
    data_column = data_parser.getColumn(iy, data_set)
    xs_units = data_column[1]
    xs = np.array(data_column[2:], dtype=np.float64)

    # parse errors
    xs_err = []
    for label in data_error_columns:

        # parse error column
        err_parser = X4ColumnParser(
            match_labels=reduce(
                lambda x, y: x + y,
                [label],
            ),
            match_units=angDistUnits + percentUnits + noUnits,
        )

        if label not in data_set.labels:
            raise ValueError(f"Subentry does not have a column called {label}")
        else:
            iyerr = [idx for idx, value in enumerate(data_set.labels) if value == label]
            if len(iyerr) > 1:
                raise ValueError(
                    f"Expected only one {label} column, found {len(iyerr)}"
                )

            err = err_parser.getColumn(iyerr[0], data_set)
            err_units = err[1]
            err_data = np.nan_to_num(np.array(err[2:], dtype=np.float64))
            # convert to same units as data
            if "PER-CENT" in err_units:
                err_data *= xs / 100
            elif err_units != xs_units:
                raise ValueError(
                    f"Attempted to extract error column {err[0]} with incompatible units"
                    f"{err_units} for data column {data_column[0]} with units {xs_units}"
                )

            xs_err.append(err_data)

    return xs, xs_err, xs_units


def parse_ex_energy(data_set):
    Ex = reduce(condenseColumn, [c.getValue(data_set) for c in energyExParserList])

    missing_Ex = np.all([a is None for a in Ex[2:]])
    Ex_units = Ex[1]

    Ex = np.array(
        Ex[2:],
        dtype=np.float64,
    )
    if missing_Ex:
        return Ex, Ex, None

    if Ex[0][-3:] == "-CM":
        raise NotImplementedError("Incident energy in CM frame!")

    Ex_err = reduce(condenseColumn, [c.getError(data_set) for c in energyExParserList])
    if Ex_err[1] != Ex_units:
        raise ValueError(
            f"Inconsistent units for Ex and Ex error: {Ex_units} and {Ex_err[1]}"
        )
    Ex_err = np.array(
        Ex_err[2:],
        dtype=np.float64,
    )

    return Ex, Ex_err, Ex_units


def parse_angle(data_set):
    # TODO handle cosine or momentum transfer
    # TODO how does this handle multiple matched entries
    angle = reduce(condenseColumn, [c.getValue(data_set) for c in angleParserList])
    if angle[1] != "degrees":
        raise ValueError(f"Cannot parse angle in units of {angle[1]}")
    if angle[0][-3:] != "-CM":
        raise NotImplementedError("Angle in lab frame!")
    angle = np.array(
        angle[2:],
        dtype=np.float64,
    )
    angle_err = reduce(condenseColumn, [c.getError(data_set) for c in angleParserList])
    missing_err = np.all([a is None for a in angle_err[2:]])
    if not missing_err:
        if angle_err[1] != "degrees":
            raise ValueError(f"Cannot parse angle error in units of {angle_err[1]}")
    angle_err = np.array(
        angle_err[2:],
        dtype=np.float64,
    )
    return angle, angle_err, "degrees"


def parse_inc_energy(data_set):
    Einc_lab = reduce(
        condenseColumn, [c.getValue(data_set) for c in incidentEnergyParserList]
    )

    Einc_units = Einc_lab[1]
    if Einc_lab[0][-3:] == "-CM":
        raise NotImplementedError("Incident energy in CM frame!")

    Einc_lab = np.array(
        Einc_lab[2:],
        dtype=np.float64,
    )

    Einc_lab_err = reduce(
        condenseColumn, [c.getError(data_set) for c in incidentEnergyParserList]
    )
    missing_err = np.all([a is None for a in Einc_lab_err[2:]])
    if not missing_err:
        if Einc_lab_err[1] != Einc_units:
            raise ValueError(
                "Inconsistent units for Einc and Einc error: "
                f"{Einc_units} and {Einc_lab_err[1]}"
            )
    Einc_lab_err = np.array(
        Einc_lab_err[2:],
        dtype=np.float64,
    )

    return Einc_lab, Einc_lab_err, Einc_units


def parse_angular_distribution(
    subentry,
    data_set,
    data_error_columns=None,
    vocal=False,
):
    r"""
    Extracts angular differential cross sections, returning incident and
    product excitation energy in MeV, angles and error in angle in degrees,
    and differential cross section and its error in mb/Sr all in a numpy array.
    """
    if vocal:
        print(
            f"Found subentry {subentry} with the following columns:\n{data_set.labels}"
        )

    if data_error_columns is None:
        data_error_columns = [b + "-ERR" for b in baseDataKeys] + dataTotalErrorKeys

    try:
        # parse angle
        angle, angle_err, angle_units = parse_angle(data_set)

        # parse energy if it's present
        Einc_lab, Einc_lab_err, Einc_units = parse_inc_energy(data_set)

        # parse excitation energy if it's present
        Ex, Ex_err, Ex_units = parse_ex_energy(data_set)

        # parse diff xs
        xs, xs_err, xs_units = parse_differential_data(
            data_set, data_error_columns=data_error_columns
        )
    except Exception as e:
        new_exception = type(e)(f"Error while parsing {subentry}: {e}")
        raise new_exception from e

    N = data_set.numrows()
    n_err_cols = len(xs_err)
    data_err = np.zeros((n_err_cols, N))
    data = np.zeros((7, N))

    data[:, :] = [
        Einc_lab,
        np.nan_to_num(Einc_lab_err),
        np.nan_to_num(Ex),
        np.nan_to_num(Ex_err),
        angle,
        np.nan_to_num(angle_err),
        xs,
    ]

    if vocal:
        if len(xs_err) == 0 or np.all([np.allclose(d, 0) for d in xs_err]):
            print(f"Warning: subentry {subentry} has no reported data errors")
            xs_err = np.zeros((n_err_cols, N))

    data_err[:, :] = np.nan_to_num(xs_err)

    return (
        data,
        data_err,
        data_error_columns,
        (angle_units, Einc_units, Ex_units, xs_units),
    )
