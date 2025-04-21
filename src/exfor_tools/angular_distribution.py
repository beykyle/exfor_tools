import numpy as np


class AngularDistribution:
    """
    Stores angular distribution with x and y errors for a given incident and
    residual excitation energy. Allows for multiple y_errs with different
    labels.

    Attributes:
        subentry (str): Subentry identifier.
        Einc (float): Incident energy.
        Einc_err (float): Error in incident energy.
        Einc_units (str): Units of incident energy.
        Ex (float): Residual excitation energy.
        Ex_err (float): Error in residual excitation energy.
        Ex_units (str): Units of residual excitation energy.
        x_units (str): Units of x (angle).
        y_units (str): Units of y.
        x (np.ndarray): Sorted angles in degrees.
        x_err (np.ndarray): Errors in angles.
        y (np.ndarray): Sorted y values.
        y_errs (list): List of y errors.
        y_err_labels (list): Labels for y errors.
        rows (int): Number of data points.
    """

    def __init__(
        self,
        subentry: str,
        x: np.ndarray,
        x_err: np.ndarray,
        y: np.ndarray,
        y_errs: list,
        y_err_labels: str,
        Einc: float,
        Einc_err: float,
        Einc_units: str,
        Ex: float,
        Ex_err: float,
        Ex_units: str,
        x_units: str,
        y_units: str,
        quantity: str,
    ):
        self.subentry = subentry
        self.Einc = Einc
        self.Einc_err = Einc_err
        self.Einc_units = Einc_units
        self.Ex = Ex
        self.Ex_err = Ex_err
        self.Ex_units = Ex_units
        self.x_units = x_units
        self.y_units = y_units

        sort_by_angle = x.argsort()
        self.x = x[sort_by_angle]
        self.x_err = x_err[sort_by_angle]
        self.y = y[sort_by_angle]
        self.y_errs = [y_err[sort_by_angle] for y_err in y_errs]
        self.y_err_labels = y_err_labels
        self.rows = self.x.shape[0]
        self.quantity = quantity

        assert (
            np.all(self.x[1:] - self.x[:-1] >= 0)
            and self.x[0] >= 0
            and self.x[-1] <= 180
        )


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
        raise ValueError("Ambiguous systematic error labels: " + ", ".join(labels))


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
        raise ValueError("Ambiguous statistical error labels:\n" + ", ".join(labels))


class AngularDistributionStatErr(AngularDistribution):
    """
    AngularDistribution with one total statistical uncertainty.

    Attributes:
        statistical_err (np.ndarray): Total statistical error.
    """

    def __init__(
        self,
        *args,
        statistical_err_labels=None,
        statistical_err_treatment="independent",
        systematic_err_labels=None,
    ):
        """
        Initialize an AngularDistributionStatErr instance.

        Args:
            *args: Variable length argument list for the base class.
            statistical_err_labels (list of str, optional): Labels for
                statistical errors. Defaults to [].
            statistical_err_treatment (str, optional): Method to treat
                statistical errors. Options are "independent" or "difference".
                Defaults to "independent".
            statistical_err_labels (list of str, optional): Labels for
                systematic errors. Defaults to []. Only relevant if
                statistical_err_labels is empty, meaning automatic parsing
                will be attempted.

        Raises:
            ValueError: If a column label in statistical_err_labels is not
            found in the subentry or if an unknown statistical_err_treatment
            is provided.
        """
        super().__init__(*args)
        if statistical_err_labels is None:
            statistical_err_labels, statistical_err_treatment = extract_staterr_labels(
                self.y_err_labels,
                expected_sys_errs=frozenset(
                    systematic_err_labels if systematic_err_labels is not None else []
                ),
            )

        self.statistical_err = np.zeros(
            (len(statistical_err_labels), self.rows), dtype=np.float64
        )

        for i, label in enumerate(statistical_err_labels):
            if label not in self.y_err_labels:
                raise ValueError(
                    f"Did not find error column label {label} in subentry {self.subentry}"
                )
            else:
                index = self.y_err_labels.index(label)
                self.statistical_err[i, :] = self.y_errs[index]

        if statistical_err_treatment == "independent":
            self.statistical_err = np.sqrt(np.sum(self.statistical_err**2, axis=0))
        elif statistical_err_treatment == "difference":
            self.statistical_err = -np.diff(self.statistical_err, axis=0)
        else:
            raise ValueError(
                f"Unknown statistical_err_treatment option: {statistical_err_treatment}"
            )

        self.y_err = self.statistical_err


class AngularDistributionSysStatErr(AngularDistributionStatErr):
    """
    AngularDistribution with a statistical uncertainty, and one or both
    of a systematic normalization or offset uncertainty

    Attributes:
        systematic_err (np.ndarray): Total systematic error.
    """

    def __init__(
        self,
        *args,
        statistical_err_labels=None,
        statistical_err_treatment="independent",
        systematic_err_labels=None,
        systematic_err_treatment="independent",
    ):
        """
        Initialize an AngularDistributionSysStatErr instance.

        Args:
            *args: Variable length argument list for the base class.
            statistical_err_labels (list of str, optional): Labels for
                statistical errors. Defaults to [].
            statistical_err_treatment (str, optional): Method to treat
                statistical errors. Options are "independent" or "difference".
                Defaults to "independent".
            systematic_err_labels (list of str, optional): Labels for
                systematic errors. Defaults to [].
            systematic_err_treatment (str, optional): Method to treat
                systematic errors. Options are "independent" or "difference".
                Defaults to "independent".

        Raises:
            ValueError: If a column label in systematic_err_labels is not
                found in the subentry, or if an unknown
                systematic_err_treatment is provided, or the systematic error
                column is non-uniform across angle.
        """
        super().__init__(
            *args,
            statistical_err_labels=statistical_err_labels,
            statistical_err_treatment=statistical_err_treatment,
            systematic_err_labels=systematic_err_labels,
        )

        if systematic_err_labels is None:
            systematic_err_labels, systematic_err_treatment = extract_syserr_labels(
                self.y_err_labels,
                expected_stat_errs=frozenset(
                    statistical_err_labels if statistical_err_labels is not None else []
                ),
            )

        self.systematic_offset_err = []
        self.systematic_norm_err = []
        self.general_systematic_err = []

        for i, label in enumerate(systematic_err_labels):
            if label not in self.y_err_labels:
                raise ValueError(
                    f"Did not find error column label {label} in subentry {self.subentry}"
                )
            else:
                index = self.y_err_labels.index(label)
                err = self.y_errs[index]
                ratio = err / self.y
                if np.allclose(ratio, ratio[0]):
                    self.systematic_norm_err.append(ratio[0])
                elif np.allclose(err, err[0]):
                    self.systematic_offset_err.append(err[0])
                else:
                    self.general_systematic_err.append(err)

        self.general_systematic_err = np.array(self.general_systematic_err)
        self.systematic_norm_err = np.array(self.systematic_norm_err)
        self.systematic_offset_err = np.array(self.systematic_offset_err)

        if systematic_err_treatment == "independent":
            self.systematic_offset_err = np.sqrt(
                np.sum(self.systematic_offset_err**2, axis=0)
            )
            self.systematic_norm_err = np.sqrt(
                np.sum(self.systematic_norm_err**2, axis=0)
            )
            self.general_systematic_err = np.sqrt(
                np.sum(self.general_systematic_err**2, axis=0)
            )
        else:
            raise ValueError(
                "Unknown systematic_err_treatment option:"
                f" {systematic_err_treatment}"
            )
