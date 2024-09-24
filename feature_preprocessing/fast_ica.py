import warnings
from ..template import FeaturePreprocessor
from sklearn.decomposition import FastICA
from utils.time_recorder import TimeRecorder

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)



class SKLFastICA(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()

                self.feature_preprocessor = FastICA(
                    algorithm="parallel",
                    whiten=False,
                    fun="logcosh",
                    random_state=self.global_params.seed
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Fast Independent Component Analysis", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default_value=100
        )
        algorithm = CategoricalHyperparameter(
            "algorithm", ["parallel", "deflation"], "parallel"
        )
        whiten = CategoricalHyperparameter("whiten", ["False", "True"], "False")
        fun = CategoricalHyperparameter("fun", ["logcosh", "exp", "cube"], "logcosh")
        cs.add([n_components, algorithm, whiten, fun])

        cs.add_condition(EqualsCondition(n_components, whiten, "True"))
        return cs
