from ..template import Regressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

class AdaboostRegressor(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="AdaBoost Regressor", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        # base_estimator = Constant(name="base_estimator", value="None")
        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True
        )
        loss = CategoricalHyperparameter(
            name="loss",
            choices=["linear", "square", "exponential"],
            default_value="linear",
        )
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=1, log=False
        )

        cs.add([n_estimators, learning_rate, loss, max_depth])
        return cs

