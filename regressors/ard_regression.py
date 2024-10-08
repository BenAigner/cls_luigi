from ..template import Regressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UnParametrizedHyperparameter,
)

class ARDRegression(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="ARD Regression", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)
        n_iter = UnParametrizedHyperparameter("n_iter", value=300)
        tol = UniformFloatHyperparameter(
            "tol", 10 ** -5, 10 ** -1, default_value=10 ** -3, log=True
        )
        alpha_1 = UniformFloatHyperparameter(
            name="alpha_1", lower=10 ** -10, upper=10 ** -3, default_value=10 ** -6
        )
        alpha_2 = UniformFloatHyperparameter(
            name="alpha_2",
            log=True,
            lower=10 ** -10,
            upper=10 ** -3,
            default_value=10 ** -6,
        )
        lambda_1 = UniformFloatHyperparameter(
            name="lambda_1",
            log=True,
            lower=10 ** -10,
            upper=10 ** -3,
            default_value=10 ** -6,
        )
        lambda_2 = UniformFloatHyperparameter(
            name="lambda_2",
            log=True,
            lower=10 ** -10,
            upper=10 ** -3,
            default_value=10 ** -6,
        )
        threshold_lambda = UniformFloatHyperparameter(
            name="threshold_lambda",
            log=True,
            lower=10 ** 3,
            upper=10 ** 5,
            default_value=10 ** 4,
        )
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")

        cs.add(
            [
                n_iter,
                tol,
                alpha_1,
                alpha_2,
                lambda_1,
                lambda_2,
                threshold_lambda,
                fit_intercept,
            ]
        )

        return cs
