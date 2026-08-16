import torch

from .simple_tsf_runner_rl import SimpleTimeSeriesForecastingRunnerRL


class NoBPRunnerRL(SimpleTimeSeriesForecastingRunnerRL):

    def backward(self, loss: torch.Tensor):
        pass
