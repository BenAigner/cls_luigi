from ..template import Regressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

class ExtraTreesRegressor(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="Extra Trees Regressor", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        criterion = CategoricalHyperparameter(
            "criterion", ["mse", "friedman_mse", "mae"]
        )
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1
        )

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )

        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="False"
        )

        cs.add(
            [
                criterion,
                max_features,
                max_depth,
                max_leaf_nodes,
                min_samples_split,
                min_samples_leaf,
                min_impurity_decrease,
                min_weight_fraction_leaf,
                bootstrap,
            ]
        )

        return cs
