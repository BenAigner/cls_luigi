from ..template import Classifier
from sklearn.naive_bayes import BernoulliNB
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

class SKLBernoulliNB(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                self.estimator = BernoulliNB(
                    alpha=1.0,
                    fit_prior=True,

                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Bernoulli Naive Bayes classifier", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)

        # the smoothing parameter is a non-negative float
        # I will limit it to 1000 and put it on a logarithmic scale. (SF)
        # Please adjust that, if you know a proper range, this is just a guess.
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1e-2, upper=100, default_value=1, log=True
        )

        fit_prior = CategoricalHyperparameter(
            name="fit_prior", choices=["True", "False"], default_value="True"
        )

        cs.add([alpha, fit_prior])
        return cs
