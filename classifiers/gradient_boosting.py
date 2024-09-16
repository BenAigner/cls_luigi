from ..template import Classifier
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

class SKLGradientBoosting(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                self.estimator = HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.1,
                    min_samples_leaf=20,
                    max_depth=None,
                    max_leaf_nodes=31,
                    max_bins=255,
                    l2_regularization=1e-10,
                    early_stopping=False,
                    tol=1e-7,
                    scoring="loss",
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                    random_state=self.global_params.seed,
                    warm_start=True
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Gradient Boosting Classifier", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)
        loss = Constant("loss", "auto")
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=200, default_value=20, log=True
        )
        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
        max_leaf_nodes = UniformIntegerHyperparameter(
            name="max_leaf_nodes", lower=3, upper=2047, default_value=31, log=True
        )
        max_bins = Constant("max_bins", 255)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization",
            lower=1e-10,
            upper=1,
            default_value=1e-10,
            log=True,
        )

        early_stop = CategoricalHyperparameter(
            name="early_stop", choices=["off", "valid", "train"], default_value="off"
        )
        tol = UnParametrizedHyperparameter(name="tol", value=1e-7)
        scoring = UnParametrizedHyperparameter(name="scoring", value="loss")
        n_iter_no_change = UniformIntegerHyperparameter(
            name="n_iter_no_change", lower=1, upper=20, default_value=10
        )
        validation_fraction = UniformFloatHyperparameter(
            name="validation_fraction", lower=0.01, upper=0.4, default_value=0.1
        )

        cs.add(
            [
                loss,
                learning_rate,
                min_samples_leaf,
                max_depth,
                max_leaf_nodes,
                max_bins,
                l2_regularization,
                early_stop,
                tol,
                scoring,
                n_iter_no_change,
                validation_fraction,
            ]
        )

        n_iter_no_change_cond = InCondition(
            n_iter_no_change, early_stop, ["valid", "train"]
        )
        validation_fraction_cond = EqualsCondition(
            validation_fraction, early_stop, "valid"
        )

        cs.add_conditions([n_iter_no_change_cond, validation_fraction_cond])