from ..template import Classifier
from sklearn.naive_bayes import GaussianNB
import warnings
from utils.time_recorder import TimeRecorder

from ConfigSpace.configuration_space import ConfigurationSpace

class SKLGaussianNaiveBayes(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                self.estimator = GaussianNB()

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()

    def get_hyperparameter_search_space(
            self, name="Gaussian Naive Bayes classifier", seed=123
    ):
        cs = ConfigurationSpace(name=name, seed=seed)