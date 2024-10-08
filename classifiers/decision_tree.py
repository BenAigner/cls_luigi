from ..template import Classifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

class SKLDecisionTree(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_original_target_values()

                num_features = self.x_train.shape[1]

                max_depth_factor = max(
                    1, int(np.round(0.5 * num_features, 0))
                )

                self.estimator = DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=max_depth_factor,  #
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_features=1.0,
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    random_state=self.global_params.seed,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Decision Tree Classifier", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini"
        )
        max_depth_factor = UniformFloatHyperparameter(
            "max_depth_factor", 0.0, 2.0, default_value=0.5
        )
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
        max_features = UnParametrizedHyperparameter("max_features", 1.0)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )

        cs.add(
            [
                criterion,
                max_features,
                max_depth_factor,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                min_impurity_decrease,
            ]
        )
        return cs
