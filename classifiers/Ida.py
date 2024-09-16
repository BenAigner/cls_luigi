from ..template import Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

class SKLLinearDiscriminantAnalysis(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_original_target_values()

                self.estimator = LinearDiscriminantAnalysis(
                    shrinkage=None,
                    solver="svd",
                    tol=1e-1,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Linear Discriminant Analysis", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)
        shrinkage = CategoricalHyperparameter(
            "shrinkage", ["None", "auto", "manual"], default_value="None"
        )
        shrinkage_factor = UniformFloatHyperparameter("shrinkage_factor", 0.0, 1.0, 0.5)
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True
        )
        cs.add([shrinkage, shrinkage_factor, tol])

        cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))