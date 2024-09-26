from ..template import Regressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
)

class LibLinear_SVR(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="Liblinear Support Vector Regression", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        loss = CategoricalHyperparameter(
            "loss",
            ["epsilon_insensitive", "squared_epsilon_insensitive"],
            default_value="squared_epsilon_insensitive",
        )
        # Random Guess
        epsilon = UniformFloatHyperparameter(
            name="epsilon", lower=0.001, upper=1, default_value=0.1, log=True
        )
        dual = Constant("dual", "False")
        # These are set ad-hoc
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True
        )
        fit_intercept = Constant("fit_intercept", "True")
        intercept_scaling = Constant("intercept_scaling", 1)

        cs.add(
            [C, loss, epsilon, dual, tol, fit_intercept, intercept_scaling]
        )

        dual_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(loss, "epsilon_insensitive"),
        )
        cs.add_forbidden_clause(dual_and_loss)

        return cs
