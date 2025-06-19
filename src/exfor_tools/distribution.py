import numpy as np
from .parsing import (
    parse_angular_distribution,
    parse_inc_energy,
    parse_ex_energy,
    unit_symbols,
)


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
        y_errs: list[np.ndarray],
        y_err_labels: list[str],
        xbounds=(-np.inf, np.inf),
        statistical_err_labels=None,
        statistical_err_treatment="independent",
        systematic_err_labels=None,
        systematic_err_treatment="independent",
    ):
        """
        Initializes the Distribution class with given parameters.

        Params:
            subentry (str): The subentry identifier.
            quantity (str): The quantity being measured.
            x_units (str): Units for the x values.
            y_units (str): Units for the y values.
            x (np.ndarray): Array of x values.
            x_err (np.ndarray): Array of x errors.
            y (np.ndarray): Array of y values.
            y_errs (list): List of y error arrays.
            y_err_labels (str): Labels for y errors.
            xbounds (tuple, optional): Bounds for x values. Defaults to (-np.inf, np.inf).
            statistical_err_labels (list, optional): Labels for statistical errors.
            statistical_err_treatment (str, optional): Treatment for statistical errors.
                Defaults to "independent".
            systematic_err_labels (list, optional): Labels for systematic errors.
            systematic_err_treatment (str, optional): Treatment for systematic errors.
                Defaults to "independent".

        Raises:
            ValueError: If a column label in systematic_err_labels is not found in the
                subentry, or if an unknown systematic_err_treatment is provided, or the
                systematic error column is non-uniform across angle.
            ValueError: If a column label in statistical_err_labels is not found in the
                subentry or if an unknown statistical_err_treatment is provided.
        """
        self.subentry = subentry
        self.quantity = quantity
        self.x_units = x_units
        self.y_units = y_units
        self.statistical_err_labels = statistical_err_labels
        self.statistical_err_treatment = statistical_err_treatment
        self.systematic_err_labels = systematic_err_labels
        self.systematic_err_treatment = systematic_err_treatment

        sort_by_angle = x.argsort()
        self.x = x[sort_by_angle]
        self.x_err = x_err[sort_by_angle]
        self.y = y[sort_by_angle]
        self.y_errs = [y_err[sort_by_angle] for y_err in y_errs]
        self.y_err_labels = y_err_labels
        self.rows = self.x.shape[0]
        if not (
            np.all(self.x[1:] - self.x[:-1] >= 0)
            and self.x[0] >= xbounds[0]
            and self.x[-1] <= xbounds[1]
        ):
            raise ValueError("Invalid x data!")
        self.set_errors()
        self.notes = []

    def set_errors(self):
        for err, label in zip(self.y_errs, self.y_err_labels):
            if np.any(err < 0):
                raise ValueError(f"negative errors under label {label}!")

        if self.statistical_err_labels is None:
            self.statistical_err_labels, self.statistical_err_treatment = (
                extract_staterr_labels(
                    self.y_err_labels,
                    expected_sys_errs=frozenset(
                        self.systematic_err_labels
                        if self.systematic_err_labels is not None
                        else []
                    ),
                )
            )

        self.statistical_err = np.zeros(
            (len(self.statistical_err_labels), self.rows), dtype=np.float64
        )

        for i, label in enumerate(self.statistical_err_labels):
            if label not in self.y_err_labels:
                raise ValueError(
                    f"Did not find error column label {label} in subentry {self.subentry}"
                )
            else:
                index = self.y_err_labels.index(label)
                self.statistical_err[i, :] = self.y_errs[index]

        if self.statistical_err_treatment == "independent":
            self.statistical_err = np.sqrt(np.sum(self.statistical_err**2, axis=0))
        elif self.statistical_err_treatment == "difference":
            self.statistical_err = -np.diff(self.statistical_err, axis=0)
        else:
            raise ValueError(
                f"Unknown statistical_err_treatment option: {self.statistical_err_treatment}"
            )

        if self.systematic_err_labels is None:
            self.systematic_err_labels, self.systematic_err_treatment = (
                extract_syserr_labels(
                    self.y_err_labels,
                    expected_stat_errs=frozenset(
                        self.statistical_err_labels
                        if self.statistical_err_labels is not None
                        else []
                    ),
                )
            )

        self.systematic_offset_err = []
        self.systematic_norm_err = []

        for i, label in enumerate(self.systematic_err_labels):
            if label not in self.y_err_labels:
                raise ValueError(
                    f"Did not find error column label {label} in subentry {self.subentry}"
                )
            else:
                index = self.y_err_labels.index(label)
                err = self.y_errs[index]
                ratio = err / self.y
                if np.allclose(err, err[0]):
                    self.systematic_offset_err.append(err)
                else:
                    self.systematic_norm_err.append(ratio)

        if self.systematic_norm_err == []:
            self.systematic_norm_err = [np.zeros((self.rows))]
        if self.systematic_offset_err == []:
            self.systematic_offset_err = [np.zeros((self.rows))]

        self.systematic_norm_err = np.array(self.systematic_norm_err)
        self.systematic_offset_err = np.array(self.systematic_offset_err)

        if self.systematic_err_treatment == "independent":
            self.systematic_offset_err = np.sqrt(
                np.sum(self.systematic_offset_err**2, axis=0)
            )
            self.systematic_norm_err = np.sqrt(
                np.sum(self.systematic_norm_err**2, axis=0)
            )
        else:
            raise ValueError(
                "Unknown systematic_err_treatment option:"
                f" {self.systematic_err_treatment}"
            )

        assert self.statistical_err.shape == (self.rows,)
        assert self.systematic_norm_err.shape == (self.rows,)
        assert self.systematic_offset_err.shape == (self.rows,)

    @classmethod
    def parse_subentry(
        cls,
        data_set,
        quantity: str,
        vocal=False,
        parsing_kwargs={},
    ):
        pass

    @classmethod
    def plot(cls):
        pass


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
        **kwargs,
    ):
        super().__init__(*args, xbounds=(0, 180), **kwargs)
        self.Einc = Einc
        self.Einc_err = Einc_err
        self.Einc_units = Einc_units
        self.Ex = Ex
        self.Ex_err = Ex_err
        self.Ex_units = Ex_units

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
                AngularDistribution(
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
                    AngularDistribution(
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
