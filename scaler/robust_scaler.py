from ..template import Scaler
from sklearn.preprocessing import RobustScaler
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from scipy import sparse
from sklearn.exceptions import NotFittedError


class SKLRobustScaler(Scaler):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self.scaler = RobustScaler(
                    copy=False,
                    quantile_range=(0.25, 0.75)
                )
                self._read_split_imputed_features()
                self.fit_transform_scaler()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="RobustScaler", seed=123
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace(name=name, seed=seed)
        q_min = UniformFloatHyperparameter("q_min", 0.001, 0.3, default_value=0.25)
        q_max = UniformFloatHyperparameter("q_max", 0.7, 0.999, default_value=0.75)
        cs.add((q_min, q_max))
        
    def get_hyperparameter_search_space(
        self, name="RobustScaler", seed=123):
        return None
