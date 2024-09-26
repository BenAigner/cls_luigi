from ..template import Regressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

class GaussianProcess(Regressor):
    abstract = False

    def get_hyperparameter_search_space(
            self, name="Gaussian Process", seed=123
    ):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1e-14, upper=1.0, default_value=1e-8, log=True
        )
        thetaL = UniformFloatHyperparameter(
            name="thetaL", lower=1e-10, upper=1e-3, default_value=1e-6, log=True
        )
        thetaU = UniformFloatHyperparameter(
            name="thetaU", lower=1.0, upper=100000, default_value=100000.0, log=True
        )

        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add([alpha, thetaL, thetaU])
        return cs
