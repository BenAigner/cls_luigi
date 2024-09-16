from ..template import FeaturePreprocessor
from sklearn.kernel_approximation import Nystroem
from utils.time_recorder import TimeRecorder
import warnings

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

class SKLNystroem(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()

                self.feature_preprocessor = Nystroem(
                    kernel="rbf",
                    gamma=1.0,
                    coef0=0,
                    degree=3,
                    n_components=100,
                    random_state=self.global_params.seed

                )

                self.x_train[self.x_train < 0] = 0.0
                self.x_test[self.x_test < 0] = 0.0

                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Nystroem kernel approximation", seed=123
    ):
        if dataset_properties is not None and (
                dataset_properties.get("sparse") is True
                or dataset_properties.get("signed") is False
        ):
            allow_chi2 = False
        else:
            allow_chi2 = True

        possible_kernels = ["poly", "rbf", "sigmoid", "cosine"]
        if allow_chi2:
            possible_kernels.append("chi2")
        kernel = CategoricalHyperparameter("kernel", possible_kernels, "rbf")
        n_components = UniformIntegerHyperparameter(
            "n_components", 50, 10000, default_value=100, log=True
        )
        gamma = UniformFloatHyperparameter(
            "gamma", 3.0517578125e-05, 8, log=True, default_value=0.1
        )
        degree = UniformIntegerHyperparameter("degree", 2, 5, 3)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)

        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add([kernel, degree, gamma, coef0, n_components])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])

        gamma_kernels = ["poly", "rbf", "sigmoid"]
        if allow_chi2:
            gamma_kernels.append("chi2")
        gamma_condition = InCondition(gamma, kernel, gamma_kernels)
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])