import periodictable

from .db import __EXFOR_DB__
from .parsing import quantity_matches


class Reaction:
    """
    Represents a nuclear reaction, including the target nucleus, projectile,
    and either the process or product of the reaction. Optionally, a residual
    nucleus can also be specified.
    """

    def __init__(self, target, projectile, process=None, product=None, residual=None):
        self.target = target
        self.projectile = projectile
        self.process = process
        self.product = product
        self.residual = residual

        if self.process is not None and self.product is not None:
            raise ValueError("Cannot specify both process and product in reaction")

        if self.process is None and self.product is None:
            raise ValueError("Must specify either process or product in reaction")

        if self.residual is not None and self.process is not None:
            raise ValueError("Cannot specify residual for a process reaction")

        if self.process is not None:
            self.reaction_latex = (
                f"{get_latex(*self.target)}({get_latex(*self.projectile)},"
                + f"{self.process.lower()})"
            )
            self.reaction_string = (
                f"{get_exfor_particle_symbol(*self.target)}"
                f"({get_exfor_particle_symbol(*self.projectile)},{self.process.lower()})"
            )
        elif self.residual is None:
            self.reaction_latex = (
                f"{get_latex(*self.target)}({get_latex(*self.projectile)},"
                + f"{get_latex(*self.product)})"
            )
            self.reaction_string = (
                f"{get_exfor_particle_symbol(*self.target)}"
                f"({get_exfor_particle_symbol(*self.projectile)},"
                f"{get_exfor_particle_symbol(*self.product)})f"
            )
        else:
            self.reaction_latex = (
                f"{get_latex(*self.target)}({get_latex(*self.projectile)},"
                + f"{get_latex(*self.product)}){get_latex(*self.residual)}"
            )
            self.reaction_string = (
                f"{get_exfor_particle_symbol(*self.target)}"
                f"({get_exfor_particle_symbol(*self.projectile)},"
                f"{get_exfor_particle_symbol(*self.product)})f"
                f"{get_exfor_particle_symbol(*self.residual)}"
            )

    def __str__(self):
        if self.process is not None:
            return (
                f"{get_exfor_particle_symbol(*self.target)}"
                f"({get_exfor_particle_symbol(*self.projectile)},{self.process})"
                f"{get_exfor_particle_symbol(*self.residual) if self.residual is not None else ''}"
            )
        elif self.product is not None:
            return (
                f"{get_exfor_particle_symbol(*self.target)}"
                f"({get_exfor_particle_symbol(*self.projectile)},{get_exfor_particle_symbol(*self.product)})f"
                f"{get_exfor_particle_symbol(*self.residual) if self.residual is not None else ''}"
            )
        else:
            raise ValueError("Could not figure out process or product from reaction")

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Reaction):
            return NotImplemented
        return (
            self.target == other.target
            and self.projectile == other.projectile
            and self.process == other.process
            and self.product == other.product
            and self.residual == other.residual
        )

    def __hash__(self):
        return hash(
            (
                self.target,
                self.projectile,
                self.process,
                self.product,
                self.residual,
            )
        )


def get_exfor_reaction_query(reaction: Reaction):
    """
    Constructs an EXFOR reaction query string based on the given reaction.

    Parameters:
        reaction (Reaction): The reaction object containing target, projectile,
            and process or product information.

    Returns:
        str: A formatted string representing the EXFOR reaction query.

    Raises:
        ValueError: If neither process nor product can be determined from the
            reaction.
    """
    projectile = get_exfor_particle_symbol(*reaction.projectile)
    if reaction.process is not None:
        prod = reaction.process.upper()
    elif reaction.product is not None:
        prod = get_exfor_particle_symbol(*reaction.product)
    else:
        raise ValueError("Could not figure out process or product from reaction")

    return f"{projectile},{prod}"


def query_for_reaction(reaction: Reaction, quantity: str):
    """
    Queries the EXFOR database for entries matching the given reaction
        and quantity.

    Parameters:
        reaction (Reaction): The reaction object to query
        quantity (str): The quantity to query

    Returns:
        list: A list of keys representing the matching entries in the EXFOR
            database.
    """
    exfor_quantity = quantity_matches[quantity][0][0]
    entries = __EXFOR_DB__.query(
        quantity=exfor_quantity,
        target=get_exfor_particle_symbol(*reaction.target),
        projectile=get_exfor_particle_symbol(*reaction.projectile),
        reaction=get_exfor_reaction_query(reaction),
    ).keys()
    return entries


def is_match(reaction: Reaction, subentry, vocal=False):
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

    # EXFOR Nat targets can show up as -3000 for some reason, so we need to check for that
    if target[0] == -3000:
        target = (0, target[1])

    if target != reaction.target or projectile != reaction.projectile:
        return False

    product = subentry.reaction[0].products[0]

    if isinstance(product, str):
        if reaction.process is None:
            return False
        if product != reaction.process.upper():
            return False
    else:
        product = (product.getA(), product.getZ())
        if product != reaction.product:
            return False

    if subentry.reaction[0].residual is None:
        return reaction.residual is None
    else:
        residual = (
            subentry.reaction[0].residual.getA(),
            subentry.reaction[0].residual.getZ(),
        )
        if reaction.residual is None and reaction.process.upper() in [
            "EL",
            "INL",
            "SCT",
        ]:
            return residual == reaction.target

        return residual == reaction.residual


def get_exfor_particle_symbol(A, Z):
    """
    Returns the EXFOR particle symbol for a given nucleus.

    Params:
        A: Mass number.
        Z: Atomic number.
    Returns:
        EXFOR particle symbol.
    """
    exfor_particle_symbols = {
        (1, 0): "N",
        (1, 1): "P",
        (2, 1): "D",
        (3, 1): "T",
        (4, 2): "A",
    }
    if (A, Z) in exfor_particle_symbols:
        return exfor_particle_symbols[(A, Z)]
    else:
        return f"{periodictable.elements[Z].symbol}-{A}"


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
        return r"\alpha"
    if A == 0:
        A = r"\text{nat}"

    if Ex is None:
        return f"^{{{A}}} \\rm{{{periodictable.elements[Z]}}}"
    else:
        ex = f"({float(Ex):1.3f})"
        return f"^{{{A}}} \\rm{{{periodictable.elements[Z]}}}({ex})"
