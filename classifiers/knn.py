from sklearn.neighbors import KNeighborsClassifier
from ..template import Classifier
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

class SKLKNearestNeighbors(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_original_target_values()

                self.estimator = KNeighborsClassifier(
                    n_neighbors=1,
                    weights="uniform",
                    p=2
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="K-Nearest Neighbor Classification", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1
        )
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform"
        )
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        cs.add([n_neighbors, weights, p])
        return cs
