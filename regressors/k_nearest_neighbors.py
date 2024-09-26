from ..template import Regressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

class KNearestNeighborsRegressor(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="K-Nearest Neighbor Classification", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1
        )
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform"
        )
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)

        cs.add([n_neighbors, weights, p])

        return cs
