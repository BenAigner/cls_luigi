from ..template import Regressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

class DecisionTree(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="Decision Tree Classifier", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        criterion = CategoricalHyperparameter(
            "criterion", ["mse", "friedman_mse", "mae"]
        )
        max_features = Constant("max_features", 1.0)
        max_depth_factor = UniformFloatHyperparameter(
            "max_depth_factor", 0.0, 2.0, default_value=0.5
        )
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )

        cs.add(
            [
                criterion,
                max_features,
                max_depth_factor,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                min_impurity_decrease,
            ]
        )

        return cs
