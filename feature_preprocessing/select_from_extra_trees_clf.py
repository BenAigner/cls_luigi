from ..template import FeaturePreprocessor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
class SKLSelectFromExtraTrees(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_original_target_values()

                estimator = ExtraTreesClassifier(
                    n_estimators=100,
                    criterion="gini",
                    max_features=0.5,
                    max_depth=None,
                    max_leaf_nodes=None,
                    min_samples_split=2,
                    min_weight_fraction_leaf=0.0,
                    min_impurity_decrease=0.0,
                    bootstrap=False,
                    random_state=self.global_params.seed,
                    oob_score=False,
                    n_jobs=self.global_params.n_jobs,
                    verbose=0
                )

                estimator.fit(self.x_train, self.y_train)
                self.feature_preprocessor = SelectFromModel(
                    estimator=estimator,
                    threshold="mean",
                    prefit=True
                )
                self.fit_transform_feature_preprocessor(x_and_y_required=True)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name= "Extra Trees Classifier Preprocessing", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        n_estimators = Constant("n_estimators", 100)
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini"
        )
        max_features = UniformFloatHyperparameter(
            "max_features", 0, 1, default_value=0.5
        )

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )

        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="False"
        )

        cs.add(
            [
                n_estimators,
                criterion,
                max_features,
                max_depth,
                max_leaf_nodes,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                min_impurity_decrease,
                bootstrap,
            ]
        )

        return cs
