from jitr.utils import mass
import periodictable

from .db import __EXFOR_DB__
from .parsing import quantity_matches


class Particle:
    """
    Represents a particle with mass, atomic number, and charge.

    Attributes:
        m0 (float): The rest mass of the particle.
        A (int): The atomic mass number of the particle.
        Z (int): The atomic number (charge) of the particle.
    """

    def __init__(self, m0: float, A: int, Z: int):
        """
        Initializes a Particle instance.

        Params:
            m0 (float): The rest mass of the particle.
            A (int): The atomic mass number of the particle.
            Z (int): The atomic number (charge) of the particle.
        """
        self.A = A
        self.Z = Z
        self.m0 = m0

    def latex(self):
        """
        Returns the LaTeX representation of the particle.

        Returns:
            str: LaTeX string.
        """
        return get_latex(self.A, self.Z)

    def exfor(self):
        """
        Returns the EXFOR particle symbol.

        Returns:
            str: EXFOR symbol.
        """
        return get_exfor_particle_symbol(self.A, self.Z)

    def __str__(self):
        """
        Returns the string representation of the particle.

        Returns:
            str: String representation.
        """
        return self.__repr__()

    def __eq__(self, other):
        """
        Checks equality with another particle.

        Params:
            other (Particle): Another particle to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if isinstance(other, Particle):
            return self.A == other.A and self.Z == other.Z
        elif isinstance(other, tuple):
            return self.A == other[0] and self.Z == other[1]
        else:
            return NotImplemented

    def __repr__(self):
        """
        Returns the symbolic representation of the particle.

        Returns:
            str: Symbolic representation.
        """
        return get_symbol(self.A, self.Z)

    def __iter__(self):
        """
        Allows unpacking of a Particle instance into (A, Z).

        Returns:
            iterator: An iterator over the atomic mass number and atomic number.
        """
        return iter((self.A, self.Z))

    @classmethod
    def parse(cls, p):
        if isinstance(p, tuple):
            return Nucleus(*p)
        elif isinstance(p, Nucleus):
            return p
        elif isinstance(p, Gamma):
            return p
        else:
            return NotImplemented


class Nucleus(Particle):
    """
    Represents a Nucleus with atomic mass number A and atomic number Z.

    Attributes:
        A (int): Atomic mass number.
        Z (int): Atomic number.
        Efn (float): Neutron Fermi energy.
        Efp (float): Proton Fermi energy.
    """

    def __init__(self, A: int, Z: int, **mass_kwargs):
        """
        Initializes a Nucleus instance.

        Params:
            A (int): Atomic mass number (must be greater than 0).
            Z (int): Atomic number (must be greater than or equal to 0).
            mass_kwargs: Additional keyword arguments for mass calculations.
        """

        if A > 0:
            m0 = mass.mass(A, Z, **mass_kwargs)[0]
        else:
            m0 = 0
        super().__init__(m0, A, Z)

        if A > 1:
            self.Efn = mass.neutron_fermi_energy(self.A, self.Z, **mass_kwargs)[0]
        else:
            self.Efn = 0

        if Z >= 1 and A > 1:
            self.Efp = mass.proton_fermi_energy(self.A, self.Z, **mass_kwargs)[0]
        else:
            self.Efp = 0

    def __add__(self, other):
        """
        Adds two particles together.

        Params:
            other (Nucleus): Another particle to add.

        Returns:
            Nucleus: New particle resulting from the addition.
        """
        if isinstance(other, Nucleus):
            return Nucleus(self.A + other.A, self.Z + other.Z)
        elif isinstance(other, tuple):
            return Nucleus(self.A + other[0], self.Z + other[1])
        return NotImplemented

    def __sub__(self, other):
        """
        Subtracts one particle from another.

        Params:
            other (Nucleus): Another particle to subtract.

        Returns:
            Nucleus: New particle resulting from the subtraction.
        """
        if isinstance(other, Nucleus):
            return Nucleus(self.A - other.A, self.Z - other.Z)
        elif isinstance(other, tuple):
            return Nucleus(self.A - other[0], self.Z - other[1])
        return NotImplemented


class Gamma(Particle):
    """
    Represents a gamma-ray

    Inherits from Particle with m0=0, A = 0, and Z = 0.
    """

    def __init__(self):
        """
        Initializes a Gamma object with m0=0, A=0, and Z=0.
        """
        super().__init__(0, 0, 0)

    def latex(self):
        return r"$\gamma$"

    def exfor(self):
        return r"G"

    def __repr__(self):
        return "gamma"

    def __sub__(self, other):
        raise ValueError()

    def __add__(self, other):
        return other


class Reaction:
    """Represents a nuclear reaction of the form A + a -> b + B.

    Attributes:
        target (Particle): The target particle.
        projectile (Particle): The projectile particle.
        product (Particle or str): The product particle or a string representing the process ('TOT', 'EL', 'INL', 'ABS', 'X').
        residual (Particle or None): The residual particle, or None for 'ABS' or 'TOT' processes.
        compound_system (Particle): The compound system formed by target and projectile.
        Q (float or None): The Q-value of the reaction.
        reaction_string (str): The string representation of the reaction.
        reaction_latex (str): The LaTeX representation of the reaction.
        exfor_symbol_reaction (str): The EXFOR symbol for the reaction.
    """

    def __init__(
        self,
        target: Particle,
        projectile: Particle,
        product: Particle = None,
        residual: Particle = None,
        process: str = None,
    ):
        """Initializes a Reaction instance.

        Args:
            target (Particle): The target particle.
            projectile (Particle): The projectile particle.
            product (Particle or str): The product particle or a string denoting the process ('TOT', 'EL', 'INL', 'ABS', 'X').
            residual (Particle or None): The residual particle, or None for 'ABS' or 'TOT' processes.

        Raises:
            ValueError: If isospin is not conserved or if invalid product/residual types are provided.
        """
        self.target = Particle.parse(target)
        self.projectile = Particle.parse(projectile)
        self.compound_system = self.target + self.projectile

        if product is not None:
            product = Particle.parse(product)
        if residual is not None:
            residual = Particle.parse(residual)

        if process:
            self.process = process.upper()
            if self.process in ["EL", "INL"]:
                if (product and product != self.projectile) or (
                    residual and residual != self.target
                ):
                    raise ValueError(
                        "Invalid scattering process reaction configuration."
                    )
                self.product = self.projectile
                self.residual = self.target
                self.Q = 0
            elif self.process == "ABS":
                if product or residual != self.compound_system:
                    raise ValueError("Invalid 'ABS' process reaction configuration.")
                self.product = None
                self.residual = self.compound_system
                self.Q = self.projectile.m0 + self.target.m0 - self.compound_system.m0
            elif self.process == "TOT":
                if product or residual:
                    raise ValueError("Invalid 'TOT' process reaction configuration")
                self.product = None
                self.residual = None
                self.Q = None
            elif self.process == "X":
                if product or not residual:
                    raise ValueError("Invalid 'X' process reaction configuration.")
                self.product = None
                self.residual = Particle.parse(residual)
                self.Q = None

            self.exfor_symbol_reaction = f"{self.projectile.exfor()},{self.process}"
            self.reaction_string = f"{self.target}({self.projectile},"
            f"{self.process.lower()}){self.residual or ''}"
            self.reaction_latex = f"{self.target.latex()}({self.projectile.latex()},"
            f"{self.process.lower()}){self.residual.latex() if self.residual else ''}"
        else:
            if not (product or residual):
                raise ValueError(
                    "Ambiguous reaction: one of product, residual, or process must be provided."
                )
            self.product = Particle.parse(product) if product else None
            self.residual = Particle.parse(residual) if residual else None

            if not self.product:
                self.product = self.compound_system - self.residual
            if not self.residual:
                self.residual = self.compound_system - self.product

            self.Q = (
                self.projectile.m0 + self.target.m0 - self.residual.m0 - self.product.m0
            )
            if (
                self.target.A + self.projectile.A != self.residual.A + self.product.A
            ) or (
                self.target.Z + self.projectile.Z != self.residual.Z + self.product.Z
            ):
                raise ValueError("Isospin not conserved in this reaction.")

            self.reaction_string = (
                f"{self.target}({self.projectile},{self.product}){self.residual}"
            )
            self.reaction_latex = f"{self.target.latex()}({self.projectile.latex()},{self.product.latex()}){self.residual.latex()}"
            self.exfor_symbol_reaction = (
                f"{self.projectile.exfor()},{self.product.exfor()}"
            )

    def is_match(self, subentry, vocal=False):
        """Checks if the reaction matches a given subentry.

        Args:
            subentry: The subentry to match against.
            vocal (bool, optional): If True, provides verbose output. Defaults to False.

        Returns:
            bool: True if the reaction matches the subentry, False otherwise.
        """
        target = (subentry.reaction[0].targ.getA(), subentry.reaction[0].targ.getZ())
        projectile = (
            subentry.reaction[0].proj.getA(),
            subentry.reaction[0].proj.getZ(),
        )

        if target != self.target or projectile != self.projectile:
            return False

        product = subentry.reaction[0].products[0]
        if isinstance(product, str):
            if product != self.process:
                return False
        else:
            product = (product.getA(), product.getZ())
            if product != self.product:
                return False

        if subentry.reaction[0].residual is None:
            return self.residual is None
        else:
            residual = (
                subentry.reaction[0].residual.getA(),
                subentry.reaction[0].residual.getZ(),
            )
            return residual == self.residual

    def query(self, quantity: str):
        """Queries the EXFOR database for entries matching the reaction and quantity.

        Args:
            quantity (str): The quantity to query.

        Returns:
            list: A list of entries matching the query.
        """
        exfor_quantity = quantity_matches[quantity][0][0]
        entries = __EXFOR_DB__.query(
            quantity=exfor_quantity,
            target=self.target.exfor(),
            projectile=self.projectile.exfor(),
            reaction=self.exfor_symbol_reaction,
        ).keys()
        return entries

    def __eq__(self, other):
        """Checks equality with another Reaction instance.

        Args:
            other (Reaction): The other Reaction instance to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, Reaction):
            return False
        return (
            self.target,
            self.projectile,
            self.product,
            self.residual,
            self.process,
        ) == (
            other.target,
            other.projectile,
            other.product,
            other.residual,
            other.process,
        )

    def __hash__(self):
        """Returns the hash of the Reaction instance.

        Returns:
            int: The hash value.
        """
        return hash((self.target, self.projectile, self.product, self.residual))


class ElasticReaction(Reaction):
    """
    Represents an elastic reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, None, None, "EL", **kwargs)


class InelasticReaction(Reaction):
    """
    Represents an inelastic reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, None, None, "INL", target, **kwargs)


class TotalReaction(Reaction):
    """
    Represents a total reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, None, None, "TOT", None, **kwargs)


class AbsorptionReaction(Reaction):
    """
    Represents an absorption reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, None, None, "ABS", None, **kwargs)


class InclusiveReaction(Reaction):
    """
    Represents an inclusive reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        residual: The residual nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, residual, **kwargs):
        super().__init__(target, projectile, None, None, "X", residual, **kwargs)


class GammaCaptureReaction(Reaction):
    """
    Represents a gamma capture reaction.

    Params:
        target: The target nucleus.
        projectile: The projectile nucleus.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, target, projectile, **kwargs):
        residual = target + projectile
        product = Gamma()
        super().__init__(target, projectile, residual, product, **kwargs)


def get_latex(A, Z, Ex=None):
    """
    Returns the LaTeX representation of a nucleus.

    Params:
        A: Mass number.
        Z: Atomic number.
        Ex: Excitation energy (optional).
    Returns:
        LaTeX string.
    """
    if (A, Z) == (1, 0):
        return "n"
    elif (A, Z) == (1, 1):
        return "p"
    elif (A, Z) == (2, 1):
        return "d"
    elif (A, Z) == (3, 1):
        return "t"
    elif (A, Z) == (4, 2):
        return r"$\alpha$"
    else:
        if Ex is None:
            return f"$^{{{A}}}${str(periodictable.elements[Z])}"
        else:
            ex = f"({float(Ex):1.3f})"
            return f"$^{{{A}}}${str(periodictable.elements[Z])}{ex}"


def get_symbol(A, Z, Ex=None):
    """
    Returns the symbol representation of a nucleus.

    Params:
        A: Mass number.
        Z: Atomic number.
        Ex: Excitation energy (optional).
    Returns:
        Symbol string.
    """
    if (A, Z) == (1, 0):
        return "n"
    elif (A, Z) == (1, 1):
        return "p"
    elif (A, Z) == (2, 1):
        return "d"
    elif (A, Z) == (3, 1):
        return "t"
    elif (A, Z) == (4, 2):
        return r"alpha"
    else:
        if Ex is None:
            return f"{A}-{str(periodictable.elements[Z])}"
        else:
            ex = f"({float(Ex):1.3f})"
            return f"{A}-{str(periodictable.elements[Z])}{ex}"


def get_exfor_particle_symbol(A, Z):
    """
    Returns the EXFOR particle symbol for a given nucleus.

    Params:
        A: Mass number.
        Z: Atomic number.
    Returns:
        EXFOR particle symbol.
    """
    return {
        (1, 0): "N",
        (1, 1): "P",
        (2, 1): "D",
        (3, 1): "T",
        (4, 2): "A",
    }.get(
        (A, Z),
        f"{str(periodictable.elements[Z])}-{A}",
    )
