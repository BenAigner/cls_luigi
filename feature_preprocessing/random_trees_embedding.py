from ..template import FeaturePreprocessor
from sklearn.ensemble import RandomTreesEmbedding
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

class SKLRandomTreesEmbedding(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()

                self.feature_preprocessor = RandomTreesEmbedding(
                    n_estimators=10,
                    max_depth=5,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0,  # TODO: check
                    max_leaf_nodes=None,
                    n_jobs=self.global_params.n_jobs,
                    random_state=self.global_params.seed,
                    sparse_output=False
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=False, handle_sparse_output=False)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Random Trees Embedding", seed=123
    ):
        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=10, upper=100, default_value=10
        )
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=2, upper=10, default_value=5
        )
        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1
        )
        min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 1.0)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None"
        )
        bootstrap = CategoricalHyperparameter("bootstrap", ["True", "False"])
        cs = ConfigurationSpace(name, seed)
        cs.add(
            [
                n_estimators,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                bootstrap,
            ]
        )
        return cs
