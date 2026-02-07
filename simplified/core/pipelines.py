from core.steps import DataStep, ModelStep, EvalStep, AnalysisStep, Pipeline
from typing import TypeVar

T = TypeVar('T')

class FullPipeline(Pipeline[T]):
    def __init__(self, data_step: DataStep, model_step: ModelStep, eval_step: EvalStep, analysis_step: AnalysisStep):
        self.data_step = data_step
        self.model_step = model_step
        self.eval_step = eval_step
        self.analysis_step = analysis_step

    def run(self) -> T:
        data = self.data_step.prepare()
        model = self.model_step.train(data)
        results = self.eval_step.evaluate(model, data)
        return self.analysis_step.analyze(data, results)

class TrainSavePipeline:
    def __init__(self, data_step: DataStep, model_step: ModelStep):
        self.data_step = data_step
        self.model_step = model_step

    def run(self) -> None:
        data = self.data_step.prepare()
        self.model_step.train(data)
        self.model_step.save()

class LoadEvaluateAnalyzePipeline(Pipeline[T]):
    def __init__(self, data_step: DataStep, model_step: ModelStep, eval_step: EvalStep, analysis_step: AnalysisStep):
        self.data_step = data_step
        self.model_step = model_step
        self.eval_step = eval_step
        self.analysis_step = analysis_step

    def run(self) -> T:
        data = self.data_step.prepare()
        model = self.model_step.load()
        results = self.eval_step.evaluate(model, data)
        return self.analysis_step.analyze(data, results)