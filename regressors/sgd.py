from ..template import Regressor

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UnParametrizedHyperparameter,
)

class SGD(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="Stochastic Gradient Descent Regressor", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        loss = CategoricalHyperparameter(
            "loss",
            [
                "squared_loss",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
            default_value="squared_loss",
        )
        penalty = CategoricalHyperparameter(
            "penalty", ["l1", "l2", "elasticnet"], default_value="l2"
        )
        alpha = UniformFloatHyperparameter(
            "alpha", 1e-7, 1e-1, log=True, default_value=0.0001
        )
        l1_ratio = UniformFloatHyperparameter(
            "l1_ratio", 1e-9, 1.0, log=True, default_value=0.15
        )
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True
        )
        epsilon = UniformFloatHyperparameter(
            "epsilon", 1e-5, 1e-1, default_value=0.1, log=True
        )
        learning_rate = CategoricalHyperparameter(
            "learning_rate",
            ["optimal", "invscaling", "constant"],
            default_value="invscaling",
        )
        eta0 = UniformFloatHyperparameter(
            "eta0", 1e-7, 1e-1, default_value=0.01, log=True
        )
        power_t = UniformFloatHyperparameter("power_t", 1e-5, 1, default_value=0.25)
        average = CategoricalHyperparameter(
            "average", ["False", "True"], default_value="False"
        )

        cs.add(
            [
                loss,
                penalty,
                alpha,
                l1_ratio,
                fit_intercept,
                tol,
                epsilon,
                learning_rate,
                eta0,
                power_t,
                average,
            ]
        )

        # TODO add passive/aggressive here, although not properly documented?
        elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
        epsilon_condition = InCondition(
            epsilon,
            loss,
            ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        )

        # eta0 is only relevant if learning_rate!='optimal' according to code
        # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
        # linear_model/sgd_fast.pyx#L603
        eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling", "constant"])
        power_t_condition = EqualsCondition(power_t, learning_rate, "invscaling")

        cs.add_conditions(
            [elasticnet, epsilon_condition, power_t_condition, eta0_in_inv_con]
        )

        return cs
