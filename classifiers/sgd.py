from ..template import Classifier
from sklearn.linear_model import SGDClassifier
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UnParametrizedHyperparameter,
)

class SKLSGD(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                self.estimator = SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=0.0001,
                    l1_ratio=0.15,
                    fit_intercept=True,
                    tol=1e-4,
                    epsilon=1e-4,
                    learning_rate="invscaling",
                    eta0=0.01,
                    power_t=0.5,
                    average=False,
                    random_state=self.global_params.seed
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Stochastic Gradient Descent Classifier", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        loss = CategoricalHyperparameter(
            "loss",
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            default_value="log",
        )
        penalty = CategoricalHyperparameter(
            "penalty", ["l1", "l2", "elasticnet"], default_value="l2"
        )
        alpha = UniformFloatHyperparameter(
            "alpha", 1e-7, 1e-1, log=True, default_value=0.0001
        )
        l1_ratio = UniformFloatHyperparameter(
            "l1_ratio", 1e-9, 1, log=True, default_value=0.15
        )
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, log=True, default_value=1e-4
        )
        epsilon = UniformFloatHyperparameter(
            "epsilon", 1e-5, 1e-1, default_value=1e-4, log=True
        )
        learning_rate = CategoricalHyperparameter(
            "learning_rate",
            ["optimal", "invscaling", "constant"],
            default_value="invscaling",
        )
        eta0 = UniformFloatHyperparameter(
            "eta0", 1e-7, 1e-1, default_value=0.01, log=True
        )
        power_t = UniformFloatHyperparameter("power_t", 1e-5, 1, default_value=0.5)
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
        epsilon_condition = EqualsCondition(epsilon, loss, "modified_huber")

        power_t_condition = EqualsCondition(power_t, learning_rate, "invscaling")

        # eta0 is only relevant if learning_rate!='optimal' according to code
        # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
        # linear_model/sgd_fast.pyx#L603
        eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling", "constant"])
        cs.add_conditions(
            [elasticnet, epsilon_condition, power_t_condition, eta0_in_inv_con]
        )