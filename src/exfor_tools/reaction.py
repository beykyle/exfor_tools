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
                f"{get_exfor_particle_symbol(*self.product)})"
            )
        else:
            self.reaction_latex = (
                f"{get_latex(*self.target)}({get_latex(*self.projectile)},"
                + f"{get_latex(*self.product)}){get_latex(*self.residual)}"
            )
            self.reaction_string = (
                f"{get_exfor_particle_symbol(*self.target)}"
                f"({get_exfor_particle_symbol(*self.projectile)},"
                f"{get_exfor_particle_symbol(*self.product)})"
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
                f"({get_exfor_particle_symbol(*self.projectile)},{get_exfor_particle_symbol(*self.product)})"
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


#: Columns by which a data set states which residual excitation it covers. Some
#: resolve it to a single level -- E-LVL, E-EXC, LVL-NUMB -- and some only bound it,
#: EXFOR spelling the bound E-LVL-MAX, E-EXC-MAX or E-EXC-MX-A. All are matched here,
#: so the fragments below are deliberately prefixes.
EXCITATION_COLUMN_FRAGMENTS = ("E-LVL", "E-EXC", "LVL-NUMB")


def specifies_excitation(subentry) -> bool:
    """Whether a data set states the residual excitation it covers.

    True both when the excitation is resolved to one level and when it is merely
    bounded, in which case the data set is summed over every level below the bound --
    the ground state plus whatever low-lying levels the experiment could not separate.
    The two are not the same thing, and this predicate deliberately does not
    distinguish them: the published nucleon-nucleus corpora count both as elastic,
    with bounds running from 30 keV on 93Nb up to 800 keV. A caller that needs the
    ground state alone must check for a resolved column itself.
    """
    return any(
        any(fragment in label for fragment in EXCITATION_COLUMN_FRAGMENTS)
        for label in subentry.labels
    )


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
        process = reaction.process.upper()
        if product != process:
            # The EXFOR dictionary defines SCT as "Total scattering (elastic +
            # inelastic)". Summed over everything, that is not elastic scattering and
            # must not satisfy a query for it. But many measurements are written as
            # (n,SCT) against an excitation column rather than being split into (n,EL)
            # and (n,INL): either resolved to a level, whose ground state *is* elastic,
            # or bounded above by a few tens to a few hundred keV, which is the ground
            # state plus the low-lying levels the experiment could not separate. Both
            # match, leaving the excitation-energy filter to select the channel where
            # the data set resolves one -- so a caller wanting only the elastic channel
            # must pass elastic_only=True, which forces Ex_range to (0, 0). A data set
            # that only bounds its excitation has no column for that filter to act on
            # and is admitted whole; see specifies_excitation.
            if not (
                process == "EL" and product == "SCT" and specifies_excitation(subentry)
            ):
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
        # the residual of a natural target carries the same -3000 sentinel as the
        # target, and must be normalized the same way, or no elastic scattering data
        # set on a natural target will ever match
        if residual[0] == -3000:
            residual = (0, residual[1])

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
