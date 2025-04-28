from jitr.reactions import (
    Reaction,
    Particle,
    Gamma,
    Electron,
    Positron,
    Nucleus,
    ElasticReaction,
    InelasticReaction,
    TotalReaction,
    AbsorptionReaction,
    InclusiveReaction,
    GammaCaptureReaction,
)
import periodictable

from .db import __EXFOR_DB__
from .parsing import quantity_matches


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

    if target != reaction.target or projectile != reaction.projectile:
        return False

    product = subentry.reaction[0].products[0]
    if isinstance(product, str):
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
