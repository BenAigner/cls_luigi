from ..template import FeaturePreprocessor
from sklearn.feature_selection import chi2, GenericUnivariateSelect
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace import NotEqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

class SKLSelectRates(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_original_target_values()

                self.feature_preprocessor = GenericUnivariateSelect(
                    score_func=chi2,
                    mode="fpr",
                    param=0.1
                )

                self.x_train[self.x_train < 0] = 0.0
                self.x_test[self.x_test < 0] = 0.0

                self.fit_transform_feature_preprocessor(x_and_y_required=True)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Univariate Feature Selection based on rates", seed=123
    ):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1
        )

        if dataset_properties is not None and dataset_properties.get("sparse"):
            choices = ["mutual_info_regression", "f_regression"]
        else:
            choices = ["f_regression"]

        score_func = CategoricalHyperparameter(
            name="score_func", choices=choices, default_value="f_regression"
        )

        mode = CategoricalHyperparameter("mode", ["fpr", "fdr", "fwe"], "fpr")

        cs = ConfigurationSpace(name=name,seed=seed)
        cs.add(alpha)
        cs.add(score_func)
        cs.add(mode)

        # Mutual info consistently crashes if percentile is not the mode
        if "mutual_info_regression" in choices:
            cond = NotEqualsCondition(mode, score_func, "mutual_info_regression")
            cs.add_condition(cond)

        return cs
