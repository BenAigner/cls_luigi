from ..template import Classifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
class SKLExtraTrees(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                max_features = int(self.x_train.shape[1] ** 0.5)
                if max_features == 0:
                    max_features = "sqrt"
                self.estimator = ExtraTreesClassifier(
                    criterion="gini",
                    max_features=max_features,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    bootstrap=False,
                    random_state=self.global_params.seed
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Extra Trees Classifier", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini"
        )

        # The maximum number of features used in the forest is calculated as
        # m^max_features, where m is the total number of features,
        # and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features
        # in the estimator. This corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0.0, 1.0, default_value=0.5
        )

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )

        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="False"
        )
        cs.add(
            [
                criterion,
                max_features,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                min_impurity_decrease,
                bootstrap,
            ]
        )
        return cs
