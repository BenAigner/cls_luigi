from ..template import FeaturePreprocessor
from sklearn.preprocessing import PolynomialFeatures
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)


class SKLPolynomialFeatures(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()

                self.feature_preprocessor = PolynomialFeatures(
                    degree=2,
                    interaction_only=False,
                    include_bias=True
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="PolynomialFeatures", seed=123
    ):
        # More than degree 3 is too expensive!
        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter(
            "interaction_only", ["False", "True"], "False"
        )
        include_bias = CategoricalHyperparameter(
            "include_bias", ["True", "False"], "True"
        )

        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add([degree, interaction_only, include_bias])
        return cs
