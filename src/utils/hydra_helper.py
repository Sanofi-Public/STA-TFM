""" hydra_helper.py
"""
import hydra


def init_hydra():
    """
    initializes hydra with config from root/conf folder
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../conf", version_base="1.1")