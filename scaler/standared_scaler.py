from ..template import Scaler
from sklearn.preprocessing import StandardScaler
import warnings
from utils.time_recorder import TimeRecorder


class SKLStandardScaler(Scaler):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self.scaler = StandardScaler(
                    copy=False
                )
                self._read_split_imputed_features()
                self.fit_transform_scaler()
                self.sava_outputs()
                
    def get_hyperparameter_search_space(
        self, name="SKLStandardScaler", seed=123):
        return None
