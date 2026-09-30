from setuptools import find_packages, setup

setup(
    name="formation_nav",
    version="0.1.0",
    description="FormationNav task (take-off, formation, obstacle-aware waypoint navigation, hold) "
    "and FC-LSTM-FC MAPPO for OmniDrones",
    packages=find_packages(include=["formation_nav", "formation_nav.*"]),
    package_data={"formation_nav": ["assets/*.yaml", "assets/OMNIDRONES_LICENSE"]},
    python_requires=">=3.10",
    # torch / torchrl / tensordict / hydra come from the OmniDrones installation
    install_requires=["matplotlib", "pyyaml"],
)
