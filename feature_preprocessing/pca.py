from ..template import FeaturePreprocessor
from sklearn.decomposition import PCA
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

class SKLPCA(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()

                self.feature_preprocessor = PCA(
                    n_components=0.9999,
                    whiten=False,
                    copy=True,
                    random_state=self.global_params.seed,
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Principle Component Analysis", seed=123
    ):
        keep_variance = UniformFloatHyperparameter(
            "keep_variance", 0.5, 0.9999, default_value=0.9999
        )
        whiten = CategoricalHyperparameter(
            "whiten", ["False", "True"], default_value="False"
        )
        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add([keep_variance, whiten])
        return cs
