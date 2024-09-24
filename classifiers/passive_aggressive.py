from sklearn.linear_model import PassiveAggressiveClassifier
from ..template import Classifier
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UnParametrizedHyperparameter,
)

class SKLPassiveAggressive(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                self.estimator = PassiveAggressiveClassifier(
                    C=1.0,
                    fit_intercept=True,
                    loss="hinge",
                    tol=1e-4,
                    average=False,
                    shuffle=True,
                    random_state=self.global_params.seed,
                    warm_start=True
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Passive Aggressive Classifier", seed=123
    ):
        C = UniformFloatHyperparameter("C", 1e-5, 10, 1.0, log=True)
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        loss = CategoricalHyperparameter(
            "loss", ["hinge", "squared_hinge"], default_value="hinge"
        )

        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True
        )
        # Note: Average could also be an Integer if > 1
        average = CategoricalHyperparameter(
            "average", ["False", "True"], default_value="False"
        )

        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add([loss, fit_intercept, tol, C, average])
        return cs
