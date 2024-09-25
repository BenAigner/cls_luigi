from ..template import Classifier
from sklearn.svm import SVC
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
)
class SKLKernelSVC(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_original_target_values()

                self.estimator = SVC(
                    C=1.0,
                    kernel="rbf",
                    degree=3,
                    gamma=0.1,
                    coef0=0.0,
                    shrinking=True,
                    tol=1e-3,
                    max_iter=-1,
                    random_state=self.global_params.seed,
                    class_weight=None,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Liblinear Support Vector Classification", seed=123):
    
        cs = ConfigurationSpace(name=name, seed=seed)

        penalty = CategoricalHyperparameter("penalty", ["l1", "l2"], default_value="l2")
        loss = CategoricalHyperparameter(
            "loss", ["hinge", "squared_hinge"], default_value="squared_hinge"
        )
        dual = Constant("dual", "False")
        # This is set ad-hoc
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True
        )
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
        multi_class = Constant("multi_class", "ovr")
        # These are set ad-hoc
        fit_intercept = Constant("fit_intercept", "True")
        intercept_scaling = Constant("intercept_scaling", 1)
        cs.add_hyperparameters(
            [penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling]
        )

        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"), ForbiddenEqualsClause(loss, "hinge")
        )
        constant_penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(penalty, "l2"),
            ForbiddenEqualsClause(loss, "hinge"),
        )
        penalty_and_dual = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"), ForbiddenEqualsClause(penalty, "l1")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        cs.add_forbidden_clause(constant_penalty_and_loss)
        cs.add_forbidden_clause(penalty_and_dual)
        
        return cs
