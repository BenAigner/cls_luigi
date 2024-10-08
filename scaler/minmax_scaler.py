from ..template import Scaler
from sklearn.preprocessing import MinMaxScaler
import warnings
from utils.time_recorder import TimeRecorder


class SKLMinMaxScaler(Scaler):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self.scaler = MinMaxScaler(
                    copy=False
                )
                self._read_split_imputed_features()
                self.fit_transform_scaler()
                self.sava_outputs()


    def get_hyperparameter_search_space(
        self, name="MinMaxScaler", seed=123):
        return None
