from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ..template import Classifier
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

class SKLQuadraticDiscriminantAnalysis(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                self.estimator = QuadraticDiscriminantAnalysis(
                    reg_param=0.0,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Quadratic Discriminant Analysis", seed=123
    ):
        reg_param = UniformFloatHyperparameter("reg_param", 0.0, 1.0, default_value=0.0)
        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add_hyperparameter(reg_param)