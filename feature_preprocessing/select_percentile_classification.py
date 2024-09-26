from ..template import FeaturePreprocessor
from sklearn.feature_selection import SelectPercentile, chi2
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
)
class SKLSelectPercentile(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_original_target_values()

                self.feature_preprocessor = SelectPercentile(
                    score_func=chi2,
                    percentile=50
                )
                self.x_train[self.x_train < 0] = 0.0
                self.x_test[self.x_test < 0] = 0.0

                self.fit_transform_feature_preprocessor(x_and_y_required=True)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Select Percentile Classification", seed=123
    ):
        percentile = UniformFloatHyperparameter(
            name="percentile", lower=1, upper=99, default_value=50
        )

        score_func = CategoricalHyperparameter(
            name="score_func",
            choices=["chi2", "f_classif", "mutual_info"],
            default_value="chi2",
        )
        if dataset_properties is not None:
            # Chi2 can handle sparse data, so we respect this
            if "sparse" in dataset_properties and dataset_properties["sparse"]:
                score_func = Constant(name="score_func", value="chi2")

        cs = ConfigurationSpace(name=name,seed=seed)
        cs.add([percentile, score_func])

        return cs
