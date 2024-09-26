from ..template import Regressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

class RandomForest(Regressor)
    abstract = False

    def get_hyperparameter_search_space(
            self, name="Random Forest Regressor", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)
        criterion = CategoricalHyperparameter(
            "criterion", ["mse", "friedman_mse", "mae"]
        )

        # In contrast to the random forest classifier we want to use more max_features
        # and therefore have this not on a sqrt scale
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1.0
        )

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True"
        )

        cs.add(
            [
                criterion,
                max_features,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                min_impurity_decrease,
                bootstrap,
            ]
        )

        return cs
