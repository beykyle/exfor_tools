from jitr.utils import mass
import periodictable

from .db import __EXFOR_DB__
from .parsing import quantity_matches


class Particle:
    """
    Represents a particle with atomic mass number and atomic number.

    :param A: Atomic mass number or a tuple of (A, Z).
    :type A: int or tuple
    :param Z: Atomic number.
    :type Z: int
    :param mass_kwargs: Additional keyword arguments for mass calculations.
    """

    def __init__(self, A: int, Z: int, **mass_kwargs):
        if isinstance(A, tuple) and len(A) == 2:
            self.A, self.Z = A
        else:
            self.A = A
            self.Z = Z

        if A > 0:
            self.m0 = mass.mass(self.A, self.Z, **mass_kwargs)[0]
        else:
            self.m0 = 0
        if A > 1:
            self.Efn = mass.neutron_fermi_energy(self.A, self.Z, **mass_kwargs)[0]
        else:
            self.Efn = 0

        if Z >= 1 and A > 1:
            self.Efp = mass.proton_fermi_energy(self.A, self.Z, **mass_kwargs)[0]
        else:
            self.Efp = 0

    def latex(self):
        """
        Returns the LaTeX representation of the particle.

        :return: LaTeX string.
        :rtype: str
        """
        return get_latex(self.A, self.Z)

    def exfor(self):
        """
        Returns the EXFOR particle symbol.

        :return: EXFOR symbol.
        :rtype: str
        """
        return get_exfor_particle_symbol(self.A, self.Z)

    def __str__(self):
        """
        Returns the string representation of the particle.

        :return: String representation.
        :rtype: str
        """
        return self.__repr__()

    def __eq__(self, other):
        """
        Checks equality with another particle.

        :param other: Another particle to compare.
        :type other: Particle
        :return: True if equal, False otherwise.
        :rtype: bool
        """
        return self.A == other.A and self.Z == other.Z

    def __repr__(self):
        """
        Returns the symbolic representation of the particle.

        :return: Symbolic representation.
        :rtype: str
        """
        return get_symbol(self.A, self.Z)

    def __add__(self, other):
        """
        Adds two particles together.

        :param other: Another particle to add.
        :type other: Particle
        :return: New particle resulting from the addition.
        :rtype: Particle
        """
        if isinstance(other, Particle):
            return Particle(self.A + other.A, self.Z + other.Z)
        return NotImplemented


class Gamma(Particle):
    """
    Represents a gamma particle.

    Inherits from Particle with A = 0 and Z = 0.
    """

    def __init__(self):
        super().__init__(0, 0)
        self.m0 = 0

    def latex(self):
        return r"$\gamma$"

    def exfor(self):
        return r"G"

    def __repr__(self):
        return "gamma"


class Reaction:
    """Represents a simple A + a -> b + B reaction.

    Attributes:
        target (Particle): The target particle.
        projectile (Particle): The projectile particle.
        product (Particle or str): The product particle or a string
            representing the process (one of 'TOT', 'EL', 'INL', 'ABS' or 'X').
        residual (Particle or None): The residual particle (or None if the
            process is 'ABS' or 'TOT').
        compound_system (Particle): The compound system formed by target and
            projectile.
        Q (float or None): The Q-value of the reaction.
        reaction_string (str): The string representation of the reaction.
        reaction_latex (str): The LaTeX representation of the reaction.
        exfor_symbol_reaction (str): The EXFOR symbol for the reaction.
    """

    def __init__(
        self,
        target: Particle,
        projectile: Particle,
        product: Particle,
        residual: Particle,
    ):
        """Initializes a Reaction instance.

        Args:
            target (Particle): The target particle.
            projectile (Particle): The projectile particle.
            product (Particle or str): The product particle or a string.
            residual (Particle or None): The residual particle or None.

        Raises:
            ValueError: If isospin is not conserved or if invalid product/
                residual types are provided.
        """
        self.target = target
        self.projectile = projectile
        self.compound_system = self.target + self.projectile
        self.product = product
        self.residual = residual

        if isinstance(product, Particle) and isinstance(residual, Particle):
            Apre = self.target.A + self.projectile.A
            Apost = self.residual.A + self.product.A
            Zpre = self.target.Z + self.projectile.Z
            Zpost = self.residual.Z + self.product.Z
            if Apre != Apost and Zpre != Zpost:
                raise ValueError("Isospin not conserved in this reaction")

            self.Q = (
                self.projectile.m0 + self.target.m0 - self.residual.m0 - self.product.m0
            )
            self.reaction_string = f"{target}({projectile},{product}){residual}"
            self.reaction_latex = f"{target.latex()}({projectile.latex()},{product.latex()}){residual.latex()}"
            self.exfor_symbol_reaction = f"{projectile.exfor()},{product.exfor()}"
        elif isinstance(product, str):
            self.exfor_symbol_reaction = f"{projectile.exfor()},{product}"
            if residual is None:
                self.reaction_string = f"{target}({projectile},{product})"
                self.reaction_latex = (
                    f"{target.latex()}({projectile.latex()},{product})"
                )
                self.Q = self.projectile.m0 + self.target.m0 - self.compound_system.m0
            elif isinstance(residual, Particle):
                self.reaction_string = f"{target}({projectile},{product}){residual}"
                self.reaction_latex = f"{target.latex()}({projectile.latex()},{product}){residual.latex()}"
                self.exfor_symbol_reaction = f"{projectile.exfor()},{product}"
                self.Q = None  # inclusive measurement
                if product != "X":
                    raise ValueError(
                        f"Invalid reaction: {projectile},{product} cannot go from {target} to {residual}"
                    )
            else:
                raise ValueError(f"Invalid residual type {type(residual)}")
        else:
            raise ValueError(f"Invalid product type {type(product)}")

    def is_match(self, subentry, vocal=False):
        """Checks if the reaction matches a given subentry.

        Args:
            subentry: The subentry to match against.
            vocal (bool, optional): If True, provides verbose output. Defaults to False.

        Returns:
            bool: True if the reaction matches the subentry, False otherwise.
        """
        target = (
            subentry.reaction[0].targ.getA(),
            subentry.reaction[0].targ.getZ(),
        )
        projectile = (
            subentry.reaction[0].proj.getA(),
            subentry.reaction[0].proj.getZ(),
        )
        if target != self.target or projectile != self.projectile:
            return False

        product = subentry.reaction[0].products[0]
        if not isinstance(product, str):
            product = (
                subentry.reaction[0].products[0].getA(),
                subentry.reaction[0].products[0].getZ(),
            )
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
            target=self.exfor_symbol_target,
            projectile=self.exfor_symbol_projectile,
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
        return (self.target, self.projectile, self.product, self.residual) == (
            other.target,
            other.projectile,
            other.product,
            other.residual,
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

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments.
    """
    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, "EL", target, **kwargs)


class InelasticReaction(Reaction):
    """
    Represents an inelastic reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments.
    """
    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, "INL", target, **kwargs)


class TotalReaction(Reaction):
    """
    Represents a total reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments.
    """
    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, "TOT", None, **kwargs)


class AbsorptionReaction(Reaction):
    """
    Represents an absorption reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments.
    """
    def __init__(self, target, projectile, **kwargs):
        super().__init__(target, projectile, "ABS", None, **kwargs)


class InclusiveReaction(Reaction):
    """
    Represents an inclusive reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param residual: The residual nucleus.
    :param kwargs: Additional keyword arguments.
    """
    def __init__(self, target, projectile, residual, **kwargs):
        super().__init__(target, projectile, "X", residual, **kwargs)


class GammaCaptureReaction(Reaction):
    """
    Represents a gamma capture reaction.

    :param target: The target nucleus.
    :param projectile: The projectile nucleus.
    :param kwargs: Additional keyword arguments.
    """
    def __init__(self, target, projectile, **kwargs):
        residual = target + projectile
        product = Gamma()
        super().__init__(target, projectile, residual, product, **kwargs)


def get_latex(A, Z, Ex=None):
    """
    Returns the LaTeX representation of a nucleus.

    :param A: Mass number.
    :param Z: Atomic number.
    :param Ex: Excitation energy (optional).
    :return: LaTeX string.
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

    :param A: Mass number.
    :param Z: Atomic number.
    :param Ex: Excitation energy (optional).
    :return: Symbol string.
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

    :param A: Mass number.
    :param Z: Atomic number.
    :return: EXFOR particle symbol.
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
