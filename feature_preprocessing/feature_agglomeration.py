from ..template import FeaturePreprocessor
from sklearn.cluster import FeatureAgglomeration
import numpy as np
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)


class SKLFeatureAgglomeration(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()

                self.feature_preprocessor = FeatureAgglomeration(
                    n_clusters=min(25, self.x_train.shape[1]),
                    metric="euclidean",
                    linkage="ward",
                    pooling_func=np.mean
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Feature Agglomeration", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)
        n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 400, 25)
        affinity = CategoricalHyperparameter(
            "affinity", ["euclidean", "manhattan", "cosine"], "euclidean"
        )
        linkage = CategoricalHyperparameter(
            "linkage", ["ward", "complete", "average"], "ward"
        )
        pooling_func = CategoricalHyperparameter(
            "pooling_func", ["mean", "median", "max"]
        )

        cs.add([n_clusters, affinity, linkage, pooling_func])

        affinity_and_linkage = ForbiddenAndConjunction(
            ForbiddenInClause(affinity, ["manhattan", "cosine"]),
            ForbiddenEqualsClause(linkage, "ward"),
        )
        cs.add_forbidden_clause(affinity_and_linkage)
        return cs
