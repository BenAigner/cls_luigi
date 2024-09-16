import warnings

from ..template import FeaturePreprocessor
from sklearn.decomposition import KernelPCA

from utils.time_recorder import TimeRecorder

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


class SKLKernelPCA(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()

                self.feature_preprocessor = KernelPCA(
                    n_components=100,
                    kernel="rbf",
                    gamma=0.1,
                    degree=3,
                    coef0=0,
                    remove_zero_eig=True,
                    random_state=self.global_params.seed,
                )
                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Kernel Principal Component Analysis", seed=123
    ):
        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default_value=100
        )
        kernel = CategoricalHyperparameter(
            "kernel", ["poly", "rbf", "sigmoid", "cosine"], "rbf"
        )
        gamma = UniformFloatHyperparameter(
            "gamma",
            3.0517578125e-05,
            8,
            log=True,
            default_value=0.01,
        )
        degree = UniformIntegerHyperparameter("degree", 2, 5, 3)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add([n_components, kernel, degree, gamma, coef0])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])