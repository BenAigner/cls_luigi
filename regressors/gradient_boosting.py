from ..template import Regressor

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

class GradientBoosting(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            fself, name="Gradient Boosting Regressor", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)
        loss = CategoricalHyperparameter(
            "loss", ["least_squares"], default_value="least_squares"
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=200, default_value=20, log=True
        )
        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
        max_leaf_nodes = UniformIntegerHyperparameter(
            name="max_leaf_nodes", lower=3, upper=2047, default_value=31, log=True
        )
        max_bins = Constant("max_bins", 255)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization",
            lower=1e-10,
            upper=1,
            default_value=1e-10,
            log=True,
        )

        early_stop = CategoricalHyperparameter(
            name="early_stop", choices=["off", "valid", "train"], default_value="off"
        )
        tol = UnParametrizedHyperparameter(name="tol", value=1e-7)
        scoring = UnParametrizedHyperparameter(name="scoring", value="loss")
        n_iter_no_change = UniformIntegerHyperparameter(
            name="n_iter_no_change", lower=1, upper=20, default_value=10
        )
        validation_fraction = UniformFloatHyperparameter(
            name="validation_fraction", lower=0.01, upper=0.4, default_value=0.1
        )

        cs.add(
            [
                loss,
                learning_rate,
                min_samples_leaf,
                max_depth,
                max_leaf_nodes,
                max_bins,
                l2_regularization,
                early_stop,
                tol,
                scoring,
                n_iter_no_change,
                validation_fraction,
            ]
        )

        n_iter_no_change_cond = InCondition(
            n_iter_no_change, early_stop, ["valid", "train"]
        )
        validation_fraction_cond = EqualsCondition(
            validation_fraction, early_stop, "valid"
        )

        cs.add_conditions([n_iter_no_change_cond, validation_fraction_cond])

        return cs
