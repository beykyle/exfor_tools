import json

import numpy as np

from .parsing import (
    parse_angular_distribution,
    parse_ex_energy,
    parse_inc_energy,
    unit_symbols,
)

data_types_json = {
    "ECS": "dXS/dA",
    "APower": "Ay",
    "ECS_Rutherford": "dXS/dRuth",
}


class Distribution:
    """
    Stores distribution with x and y errors for a given incident and
    residual excitation energy. Allows for multiple y_errs with different
    labels. Attempts to parse statistical and systematic errors from those
    labels. Provides functions for updating or adjusting the distribution,
    e.g. to handle transcription errors, renormalize, or remove outliers,
    which record a record of any edits for posteriority.

    Attributes:
        subentry (str): The subentry identifier.
        quantity (str): The quantity being measured.
        x_units (str): Units for the x values.
        y_units (str): Units for the y values.
        x (np.ndarray): Array of x values.
        x_err (np.ndarray): Array of x errors.
        y (np.ndarray): Array of y values.
        y_errs (list): List of y error arrays.
        y_err_labels (str): Labels for y errors.
        rows (int): Number of data points.
        statistical_err (np.ndarray): Statistical errors.
        systematic_offset_err (np.ndarray): Systematic offset errors.
        systematic_norm_err (np.ndarray): Systematic normalization errors.
    """

    def __init__(
        self,
        subentry: str,
        quantity: str,
        x_units: str,
        y_units: str,
        x: np.ndarray,
        x_err: np.ndarray,
        y: np.ndarray,
        statistical_err: np.ndarray,
        systematic_norm_err: np.ndarray,
        systematic_offset_err: np.ndarray,
        xbounds=(-np.inf, np.inf),
    ):
        """
        Initializes the Distribution class with given parameters.
        Parameters:
        -----------
            subentry: str
                The subentry identifier.
            quantity: str
                The quantity being measured.
            x_units: str
                Units for the x values.
            y_units: str
                Units for the y values.
            x: np.ndarray
                Array of x values.
            x_err: np.ndarray
                Array of x errors.
            y: np.ndarray
                Array of y values.
            statistical_err: np.ndarray
                Statistical errors.
            systematic_norm_err: np.ndarray
                Systematic normalization errors.
            systematic_offset_err: np.ndarray
                Systematic offset errors.
            xbounds: tuple, optional
                Bounds for x values. Defaults to (-np.inf, np.inf).
        """
        self.subentry = subentry
        self.quantity = quantity
        self.x_units = x_units
        self.y_units = y_units

        sort_by_x = x.argsort()
        self.x = x[sort_by_x]
        self.x_err = x_err[sort_by_x]
        self.y = y[sort_by_x]

        self.rows = self.x.shape[0]
        if not (
            np.all(self.x[1:] - self.x[:-1] >= 0)
            and self.x[0] >= xbounds[0]
            and self.x[-1] <= xbounds[1]
        ):
            raise ValueError("Invalid x data!")

        self.notes = []

        self.statistical_err = statistical_err[sort_by_x]
        self.systematic_norm_err = systematic_norm_err[sort_by_x]
        self.systematic_offset_err = systematic_offset_err[sort_by_x]

    @classmethod
    def parse_errs_and_init(
        cls,
        subentry: str,
        quantity: str,
        x_units: str,
        y_units: str,
        x: np.ndarray,
        x_err: np.ndarray,
        y: np.ndarray,
        y_errs: list[np.ndarray],
        y_err_labels: list[str],
        xbounds=(-np.inf, np.inf),
        statistical_err_labels=None,
        statistical_err_treatment="independent",
        systematic_err_labels=None,
        systematic_err_treatment="independent",
    ):
        """
        Attempts to construct a Distribution object from the given parameters. Given
        a list of y errors with arbitrary labels, attempts to categorize them into
        statistical and systematic errors, and returns a Distribution object with
        those errors categorized.

        Parameters:
        ----------
            subentry: str
                The subentry identifier.
            quantity: str
                The quantity being measured.
            x_units: str
                Units for the x values.
            y_units: str
                Units for the y values.
            x: np.ndarray
                Array of x values.
            x_err: np.ndarray
                Array of x errors.
            y: np.ndarray
                Array of y values.
            y_errs: list[np.ndarray]
                List of arrays containing y errors.
            y_err_labels: list[str]
                Labels corresponding to each array in y_errs.
            xbounds: tuple, optional
                Bounds for x values. Defaults to (-np.inf, np.inf).
            statistical_err_labels: list[str], optional
                Labels for statistical errors. If None, will be determined from y_err_labels.
            statistical_err_treatment: str, optional
                Method to treat statistical errors. Defaults to "independent".
            systematic_err_labels: list[str], optional
                Labels for systematic errors. If None, will be determined from y_err_labels.
            systematic_err_treatment: str, optional
                Method to treat systematic errors. Defaults to "independent".

        Returns:
        ----------
            Distribution :
                A Distribution object with categorized errors.

        Raises:
        ----------
            ValueError: If a column label in systematic_err_labels is not found in the
                subentry, or if an unknown systematic_err_treatment is provided, or the
                systematic error column is non-uniform across angle.
            ValueError: If a column label in statistical_err_labels is not found in the
                subentry or if an unknown statistical_err_treatment is provided.
        """
        sort_by_x = x.argsort()
        y_errs = [y_err[sort_by_x] for y_err in y_errs]
        (
            statistical_err,
            systematic_norm_err,
            systematic_offset_err,
            statistical_err_labels,
            systematic_err_labels,
        ) = cls.determine_error_categories(
            subentry,
            len(x),
            y[sort_by_x],
            y_errs,
            y_err_labels,
            statistical_err_labels,
            statistical_err_treatment,
            systematic_err_labels,
            systematic_err_treatment,
        )
        return cls(
            subentry,
            quantity,
            x_units,
            y_units,
            x[sort_by_x],
            x_err[sort_by_x],
            y[sort_by_x],
            statistical_err,
            systematic_norm_err,
            systematic_offset_err,
            xbounds=xbounds,
        )

    @staticmethod
    def determine_error_categories(
        subentry,
        rows,
        y,
        y_errs,
        y_err_labels,
        statistical_err_labels,
        statistical_err_treatment,
        systematic_err_labels,
        systematic_err_treatment,
    ):
        if statistical_err_labels is None:
            statistical_err_labels, statistical_err_treatment = extract_staterr_labels(
                y_err_labels,
                expected_sys_errs=frozenset(
                    systematic_err_labels if systematic_err_labels is not None else []
                ),
            )

        statistical_err = []

        for i, label in enumerate(statistical_err_labels):
            if label in y_err_labels:
                index = y_err_labels.index(label)
                statistical_err.append(y_errs[index])

        if statistical_err == []:
            statistical_err = [np.zeros((rows))]
        statistical_err = np.array(statistical_err)

        if statistical_err_treatment == "independent":
            statistical_err = np.sqrt(np.sum(statistical_err**2, axis=0))
        elif statistical_err_treatment == "difference":
            statistical_err = -np.diff(statistical_err, axis=0)[0, :]
        elif statistical_err_treatment == "sum":
            statistical_err = np.sum(statistical_err, axis=0)
        else:
            raise ValueError(
                f"Unknown statistical_err_treatment option: {statistical_err_treatment}"
            )

        if np.any(statistical_err < 0):
            raise ValueError(
                f"Negative statistical error found in subentry {subentry}!"
            )

        if systematic_err_labels is None:
            systematic_err_labels, systematic_err_treatment = extract_syserr_labels(
                y_err_labels,
                expected_stat_errs=frozenset(
                    statistical_err_labels if statistical_err_labels is not None else []
                ),
            )

        systematic_offset_err = []
        systematic_norm_err = []

        for i, label in enumerate(systematic_err_labels):
            if label in y_err_labels:
                index = y_err_labels.index(label)
                err = y_errs[index]
                ratio = err / y
                if np.allclose(err, err[0]):
                    systematic_offset_err.append(err)
                else:
                    systematic_norm_err.append(ratio)

        if systematic_norm_err == []:
            systematic_norm_err = [np.zeros((rows))]
        if systematic_offset_err == []:
            systematic_offset_err = [np.zeros((rows))]

        systematic_norm_err = np.array(systematic_norm_err)
        systematic_offset_err = np.array(systematic_offset_err)

        if systematic_err_treatment == "independent":
            systematic_offset_err = np.sqrt(np.sum(systematic_offset_err**2, axis=0))
            systematic_norm_err = np.sqrt(np.sum(systematic_norm_err**2, axis=0))
        else:
            raise ValueError(
                f"Unknown systematic_err_treatment option: {systematic_err_treatment}"
            )

        assert statistical_err.shape == (rows,)
        assert systematic_norm_err.shape == (rows,)
        assert systematic_offset_err.shape == (rows,)

        return (
            statistical_err,
            systematic_norm_err,
            systematic_offset_err,
            statistical_err_labels,
            systematic_err_labels,
        )

    @classmethod
    def plot(cls):
        """
        Plots the distribution on the given axis.
        """
        raise NotImplementedError(
            "Plotting not implemented for base Distribution class."
        )


class AngularDistribution(Distribution):
    """
    Represents a quantity as a function of angle, at given incident lab
    energy and residual excitation energy
    """

    def __init__(
        self,
        Einc: float,
        Einc_err: float,
        Einc_units: str,
        Ex: float,
        Ex_err: float,
        Ex_units: str,
        *args,
    ):
        super().__init__(*args, xbounds=(0, 180))
        self.Einc = Einc
        self.Einc_err = Einc_err
        self.Einc_units = Einc_units
        self.Ex = Ex
        self.Ex_err = Ex_err
        self.Ex_units = Ex_units

    @classmethod
    def parse_errs_and_init(
        cls,
        Einc: float,
        Einc_err: float,
        Einc_units: str,
        Ex: float,
        Ex_err: float,
        Ex_units: str,
        *args,
        **kwargs,
    ):
        d = Distribution.parse_errs_and_init(
            *args,
            xbounds=(0, 180),
            **kwargs,
        )
        return cls(
            Einc,
            Einc_err,
            Einc_units,
            Ex,
            Ex_err,
            Ex_units,
            d.subentry,
            d.quantity,
            d.x_units,
            d.y_units,
            d.x,
            d.x_err,
            d.y,
            d.statistical_err,
            d.systematic_norm_err,
            d.systematic_offset_err,
        )

    @classmethod
    def parse_subentry(
        cls,
        subentry,
        data_set,
        quantity: str,
        parsing_kwargs={},
        Einc_range=(0, np.inf),
        Ex_range=(0, np.inf),
        elastic_only=False,
        vocal=False,
    ):
        r"""unrolls subentry into individual AngularDistributions for each energy"""

        Einc = parse_inc_energy(data_set)[0]
        Ex = np.nan_to_num(parse_ex_energy(data_set)[0])
        if not np.any(
            np.logical_and(
                np.logical_and(Einc >= Einc_range[0], Einc <= Einc_range[1]),
                np.logical_and(Ex >= Ex_range[0], Ex <= Ex_range[1]),
            )
        ):
            return []

        lbl_frags_to_skip = ["ANG", "EN", "E-LVL", "E-EXC"]
        err_labels = [
            label
            for label in data_set.labels
            if "ERR" in label
            and np.all([frag not in label for frag in lbl_frags_to_skip])
        ]

        data, data_err, error_columns, units = parse_angular_distribution(
            subentry,
            data_set,
            data_error_columns=err_labels,
            vocal=vocal,
        )

        measurements = sort_subentry_data_by_energy(
            subentry,
            data,
            data_err,
            error_columns,
            Einc_range,
            Ex_range,
            elastic_only,
            units,
            quantity,
            parsing_kwargs,
        )
        return measurements

    @classmethod
    def plot(
        cls,
        measurements,
        ax,
        offsets=None,
        data_symbol="",
        rxn_label="",
        log=True,
        draw_baseline=False,
        baseline_offset=None,
        xlim=[0, 180],
        fontsize=10,
        label_kwargs={
            "label_offset_factor": 2,
            "label_energy_err": False,
            "label_offset": True,
        },
    ):
        r"""
        Given a collection of measurements, plots them on the same axis with offsets
        """
        # if offsets is not a sequence, figure it out
        # TODO do the same for label_offset_factor
        if isinstance(offsets, float) or isinstance(offsets, int) or offsets is None:
            if offsets is None:
                constant_factor = 1 if log else 0
            else:
                constant_factor = offsets
            if log:
                offsets = constant_factor ** np.arange(0, len(measurements))
            else:
                offsets = constant_factor * np.arange(0, len(measurements))

        # plot each measurement and add a label
        for offset, m in zip(offsets, measurements):
            if not isinstance(m, list):
                m = [m]

            c = []
            for measurement in m:
                x = np.copy(measurement.x)
                y = np.copy(measurement.y)
                color = plot_errorbar(
                    ax,
                    np.copy(measurement.x),
                    np.copy(measurement.x_err),
                    np.copy(measurement.y),
                    np.copy(measurement.statistical_err),
                    offset,
                    log,
                )
                c.append(color)

            if draw_baseline:
                if log:
                    baseline_offset = (
                        baseline_offset if baseline_offset is not None else 1
                    )
                    baseline_height = offset * baseline_offset
                else:
                    baseline_offset = (
                        baseline_offset if baseline_offset is not None else 0
                    )
                    baseline_height = offset + baseline_offset
                ax.plot([0, 180], [baseline_height, baseline_height], "k--", alpha=0.25)

            if label_kwargs is not None:
                set_label(ax, m, c, offset, x, y, log, fontsize, **label_kwargs)

        if isinstance(measurements[0], list):
            x_units = unit_symbols.get(
                measurements[0][0].x_units, measurements[0][0].x_units
            )
            y_units = unit_symbols.get(
                measurements[0][0].y_units, measurements[0][0].y_units
            )
        else:
            x_units = unit_symbols.get(measurements[0].x_units, measurements[0].x_units)
            y_units = unit_symbols.get(measurements[0].y_units, measurements[0].y_units)

        ax.set_xlabel(r"$\theta$ [{}]".format(x_units))
        ax.set_ylabel(r"{} [{}]".format(data_symbol, y_units))
        ax.set_xticks(np.arange(0, 180.01, 30))
        if log:
            ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title(f"{rxn_label}")

        if log:
            ax.set_yscale("log")

        return offsets

    def to_json(self, citation: str = "") -> str:
        data = {
            "type": next(
                (k for k, v in data_types_json.items() if v == self.quantity), "unknown"
            ),
            "energy": float(self.Einc),
            "energy-err": float(self.Einc_err),
            "ex-energy": float(self.Ex),
            "ex-energy-err": float(self.Ex_err),
            "ex-energy-units": self.Ex_units,
            "energy-units": self.Einc_units,
            "EXFORAccessionNumber": self.subentry,
            "source": citation,
            "data": {
                "angle": self.x.tolist(),
                "angle-units": self.x_units,
                "angle-err": self.x_err.tolist(),
                "cs": self.y.tolist(),
                "cs-units": self.y_units,
                "cs-err": self.statistical_err.tolist(),
                "systematic_normalization_error": self.systematic_norm_err.tolist(),
                "systematic_offset_error": self.systematic_offset_err.tolist(),
            },
        }
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, json_file):
        data = json.load(json_file)
        measurements = []
        for measurement in data:
            subentry = measurement["EXFORAccessionNumber"]
            quantity = data_types_json.get(measurement["type"], "unknown")
            x_units = measurement["data"]["angle-units"]
            y_units = measurement["data"]["cs-units"]
            x = np.array(measurement["data"]["angle"])
            x_err = np.array(measurement["data"].get("angle-err", np.zeros_like(x)))
            y = np.array(measurement["data"]["cs"])
            statistical_err = np.array(measurement["data"]["cs-err"])
            systematic_norm_err = np.array(
                measurement["data"].get(
                    "systematic_normalization_error", np.zeros_like(y)
                )
            )
            systematic_offset_err = np.array(
                measurement["data"].get("systematic_offset_error", np.zeros_like(y))
            )

            measurements.append(
                AngularDistribution(
                    measurement["energy"],
                    measurement.get("energy-err", 0.0),
                    measurement["energy-units"],
                    measurement.get("ex-energy", 0.0),
                    measurement.get("ex-energy-err", 0.0),
                    measurement.get("ex-energy-units", "MeV"),
                    subentry,
                    quantity,
                    x_units,
                    y_units,
                    x,
                    x_err,
                    y,
                    statistical_err,
                    systematic_norm_err,
                    systematic_offset_err,
                )
            )
        return measurements


def extract_syserr_labels(
    labels,
    allowed_sys_errs=frozenset(["ERR-SYS"]),
    allowed_stat_errs=frozenset(["DATA-ERR", "ERR-T", "ERR-S"]),
    expected_stat_errs=frozenset([]),
):
    """
    Extracts systematic error labels from a list of labels.

    Parameters:
    labels (list): A list of error labels.
    allowed_sys_errs (set): A set of allowed systematic error labels.
    allowed_stat_errs (set): A set of allowed statistical error labels.

    Returns:
    tuple: A tuple containing the systematic error labels and a string specifying
    the treatment

    Raises:
    ValueError: If the statistical error labels are ambiguous
    """
    allowed_sys_err_combos = frozenset([frozenset([l]) for l in allowed_sys_errs])
    sys_err_labels = (
        frozenset(labels)
        - allowed_stat_errs
        - frozenset(["ERR-DIG"])
        - expected_stat_errs
    )
    if len(sys_err_labels) == 0:
        return [], "independent"
    if sys_err_labels in allowed_sys_err_combos:
        return list(sys_err_labels), "independent"
    else:
        labels = ", ".join(labels)
        raise ValueError(f"Ambiguous systematic error labels:\n{labels}")


def extract_staterr_labels(
    labels,
    allowed_sys_errs=frozenset(["ERR-SYS"]),
    allowed_stat_errs=frozenset(["DATA-ERR", "ERR-T", "ERR-S"]),
    expected_sys_errs=frozenset([]),
):
    """
    Extracts statistical error labels from a list of labels.

    Parameters:
    labels (list): A list of error labels.
    allowed_sys_errs (set): A set of allowed systematic error labels.
    allowed_stat_errs (set): A set of allowed statistical error labels.

    Returns:
    tuple: A tuple containing the statistical error labels and a string specifying
    the treatment

    Raises:
    ValueError: If the statistical error labels are ambiguous
    """
    if "+ERR-T" in labels and "-ERR-T" in labels:
        return ["+ERR-T", "-ERR-T"], "difference"
    if "+DATA-ERR" in labels and "+DATA-ERR" in labels:
        return ["+DATA-ERR", "-DATA-ERR"], "difference"
    allowed_stat_err_combos = set(
        [frozenset([l, "ERR-DIG"]) for l in allowed_stat_errs]
        + [frozenset([l]) for l in allowed_stat_errs | frozenset(["ERR-DIG"])]
    )
    stat_err_labels = frozenset(labels) - allowed_sys_errs - expected_sys_errs
    if len(stat_err_labels) == 0:
        return [], "independent"
    if stat_err_labels in allowed_stat_err_combos:
        return list(stat_err_labels), "independent"
    else:
        labels = ", ".join(labels)
        raise ValueError(f"Ambiguous statistical error labels:\n{labels}")


def sort_subentry_data_by_energy(
    subentry,
    data,
    data_err,
    error_columns,
    Einc_range,
    Ex_range,
    elastic_only,
    units,
    quantity,
    parsing_kwargs,
):
    angle_units, Einc_units, Ex_units, xs_units = units
    Einc_mask = np.logical_and(
        data[0, :] >= Einc_range[0],
        data[0, :] <= Einc_range[1],
    )
    data = data[:, Einc_mask]
    data_err = data_err[:, Einc_mask]

    if not elastic_only:
        Ex_mask = np.logical_and(
            data[2, :] >= Ex_range[0],
            data[2, :] <= Ex_range[1],
        )
        data = data[:, Ex_mask]
        data_err = data_err[:, Ex_mask]

    # AngularDistribution objects sorted by incident energy,
    # then excitation energy or just incident enrgy if
    # elastic_only is True
    measurements = []

    # find set of unique incident energies
    unique_Einc = np.unique(data[0, :])

    # sort and fragment data by unique incident energy
    for Einc in np.sort(unique_Einc):
        mask = np.isclose(data[0, :], Einc)
        Einc_err = data[1, mask][0]

        if elastic_only:
            measurements.append(
                AngularDistribution.parse_errs_and_init(
                    Einc,
                    Einc_err,
                    Einc_units,
                    0,
                    0,
                    Ex_units,
                    subentry,
                    quantity,
                    angle_units,
                    xs_units,
                    data[4, mask],
                    data[5, mask],
                    data[6, mask],
                    [data_err[i, mask] for i in range(data_err.shape[0])],
                    error_columns,
                    **parsing_kwargs,
                )
            )
        else:
            subset = data[2:, mask]
            subset_err = data_err[:, mask]

            # find set of unique residual excitation energies
            unique_Ex = np.unique(subset[0, :])

            # sort and fragment data by unique excitation energy
            for Ex in np.sort(unique_Ex):
                mask = np.isclose(subset[0, :], Ex)
                Ex_err = subset[1, mask][0]
                measurements.append(
                    AngularDistribution.parse_errs_and_init(
                        Einc,
                        Einc_err,
                        Einc_units,
                        0,
                        0,
                        Ex_units,
                        subentry,
                        quantity,
                        angle_units,
                        xs_units,
                        subset[2, mask],
                        subset[3, mask],
                        subset[4, mask],
                        [subset_err[i, mask] for i in range(data_err.shape[0])],
                        error_columns,
                        **parsing_kwargs,
                    )
                )
    return measurements


def set_label(
    ax,
    measurements: list,
    colors: list,
    offset,
    x,
    y,
    log,
    fontsize=10,
    label_xloc_deg=None,
    label_offset_factor=2,
    label_energy_err=False,
    label_offset=True,
    label_incident_energy=True,
    label_excitation_energy=False,
    label_exfor=False,
):
    if label_xloc_deg is None:
        if x[-1] < 60:
            label_xloc_deg = 65
        elif x[-1] < 90:
            label_xloc_deg = 95
        elif x[-1] < 120:
            label_xloc_deg = 125
        elif x[0] > 30 and x[-1] > 150:
            label_xloc_deg = 1
        elif x[0] > 20 and x[-1] > 150:
            label_xloc_deg = -18
        elif x[-1] < 150:
            label_xloc_deg = 155
        else:
            label_xloc_deg = 175

    label_yloc = offset
    if log:
        label_yloc *= label_offset_factor
    else:
        label_yloc += label_offset_factor

    label_location = (label_xloc_deg, label_yloc)

    if log:
        offset_text = f"\n($\\times$ {offset:1.0e})"
    else:
        offset_text = f"\n($+$ {offset:1.0f})"

    m = measurements[0]
    label = ""
    if label_incident_energy:
        label += f"\n{m.Einc:1.2f}"
        if label_energy_err:
            label += f" $\pm$ {m.Einc_err:1.2f}"
        label += f" {m.Einc_units}"
    if label_excitation_energy:
        label += f"\n$E_{{x}} = ${m.Ex:1.2f}"
        if label_energy_err:
            label += f" $\pm$ {m.Ex_err:1.2f}"
        label += f" {m.Ex_units}"
    if label_exfor:
        label += "\n"
        for i, m in enumerate(measurements):
            if i == len(measurements) - 1:
                label += f"{m.subentry}"
            else:
                label += f"{m.subentry},\n"
    if label_offset:
        label += offset_text

    ax.text(*label_location, label, fontsize=fontsize, color=colors[-1])


def plot_errorbar(ax, x, x_err, y, y_err, offset, log):
    if log:
        y *= offset
        y_err *= offset
    else:
        y += offset

    p = ax.errorbar(
        x,
        y,
        yerr=y_err,
        xerr=x_err,
        marker="s",
        markersize=2,
        alpha=0.75,
        linestyle="none",
        elinewidth=3,
        # capthick=2,
        # capsize=1,
    )
    return p.lines[0].get_color()
