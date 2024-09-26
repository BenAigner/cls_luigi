from ..template import FeaturePreprocessor
from sklearn.kernel_approximation import RBFSampler
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

class SKLRBFSampler(FeaturePreprocessor):
    # aka kitchen sinks
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()

                self.feature_preprocessor = RBFSampler(
                    gamma=1.0,
                    n_components=100,
                    random_state=self.global_params.seed
                )
                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Random Kitchen Sinks", seed=123
    ):
        gamma = UniformFloatHyperparameter(
            "gamma", 3.0517578125e-05, 8, default_value=1.0, log=True
        )
        n_components = UniformIntegerHyperparameter(
            "n_components", 50, 10000, default_value=100, log=True
        )
        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add([gamma, n_components])
        return cs
