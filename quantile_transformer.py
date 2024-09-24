from ..template import Scaler
from sklearn.preprocessing import QuantileTransformer
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)
class SKLQuantileTransformer(Scaler):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self.scaler = QuantileTransformer(
                    copy=False,
                    n_quantiles=1000,
                    output_distribution="uniform",
                    random_state=self.global_params.seed
                )
                self._read_split_imputed_features()
                self.fit_transform_scaler()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="QuantileTransformer", seed=123
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace(name=name, seed=seed)
        # TODO parametrize like the Random Forest as n_quantiles = n_features^param
        n_quantiles = UniformIntegerHyperparameter(
            "n_quantiles", lower=10, upper=2000, default_value=1000
        )
        output_distribution = CategoricalHyperparameter(
            "output_distribution", ["normal", "uniform"]
        )
        cs.add((n_quantiles, output_distribution))
        return cs
