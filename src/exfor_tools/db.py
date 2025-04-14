from x4i3 import exfor_manager

__EXFOR_DB__ = None


def init_exfor_db():
    """
    Initialize the EXFOR database.

    This function sets up the global EXFOR database manager if it has
    not been initialized yet.
    """
    global __EXFOR_DB__
    if __EXFOR_DB__ is None:
        __EXFOR_DB__ = exfor_manager.X4DBManagerDefault()


init_exfor_db()
