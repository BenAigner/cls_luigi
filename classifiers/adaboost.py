import warnings

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..template import Classifier
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

class SKLAdaBoost(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:

                self._read_split_original_target_values()
                self._read_split_processed_features()

                base_estimator = DecisionTreeClassifier(max_depth=1, random_state=self.global_params.seed)

                self.estimator = AdaBoostClassifier(
                    estimator=base_estimator,
                    n_estimators=50,
                    learning_rate=1.0,
                    algorithm="SAMME.R",
                    random_state=self.global_params.seed
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(self, name="AdaBoostClassifier", seed=123):
        cs = ConfigurationSpace(name=name, seed=seed)

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True
        )
        algorithm = CategoricalHyperparameter(
            name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R"
        )
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=1, log=False
        )

        cs.add([n_estimators, learning_rate, algorithm, max_depth])
        return cs
