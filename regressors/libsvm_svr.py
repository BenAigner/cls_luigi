from ..template import Regressor

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

class LibSVM_SVR(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="Support Vector Regression", seed=123
    ):
        C = UniformFloatHyperparameter(
            name="C", lower=0.03125, upper=32768, log=True, default_value=1.0
        )
        # Random Guess
        epsilon = UniformFloatHyperparameter(
            name="epsilon", lower=0.001, upper=1, default_value=0.1, log=True
        )

        kernel = CategoricalHyperparameter(
            name="kernel",
            choices=["linear", "poly", "rbf", "sigmoid"],
            default_value="rbf",
        )
        degree = UniformIntegerHyperparameter(
            name="degree", lower=2, upper=5, default_value=3
        )

        gamma = UniformFloatHyperparameter(
            name="gamma", lower=3.0517578125e-05, upper=8, log=True, default_value=0.1
        )

        # TODO this is totally ad-hoc
        coef0 = UniformFloatHyperparameter(
            name="coef0", lower=-1, upper=1, default_value=0
        )
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter(
            name="shrinking", choices=["True", "False"], default_value="True"
        )
        tol = UniformFloatHyperparameter(
            name="tol", lower=1e-5, upper=1e-1, default_value=1e-3, log=True
        )
        max_iter = UnParametrizedHyperparameter("max_iter", -1)

        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add(
            [C, kernel, degree, gamma, coef0, shrinking, tol, max_iter, epsilon]
        )

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        gamma_depends_on_kernel = InCondition(
            child=gamma, parent=kernel, values=("poly", "rbf")
        )
        coef0_depends_on_kernel = InCondition(
            child=coef0, parent=kernel, values=("poly", "sigmoid")
        )
        cs.add_conditions(
            [degree_depends_on_poly, gamma_depends_on_kernel, coef0_depends_on_kernel]
        )

        return cs
