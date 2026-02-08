from typing import Generic, TypeVar
from abc import ABC, abstractmethod

T = TypeVar('T')  # Data type
M = TypeVar('M')  # Model type
R = TypeVar('R')  # Evaluation result type
C = TypeVar('C')  # Analysis result type
P = TypeVar('P')  # Pipeline result type

class DataStep(Generic[T], ABC):
    @abstractmethod
    def prepare(self) -> T:
        raise NotImplementedError

class ModelStep(Generic[M, T], ABC):
    @abstractmethod
    def train(self, data: T) -> M:
        raise NotImplementedError
    
    def save(self) -> None:
        raise NotImplementedError
    
    def load(self) -> M:
        raise NotImplementedError

class EvalStep(Generic[M, T, R], ABC):
    @abstractmethod
    def evaluate(self, model: M, data: T) -> R:
        raise NotImplementedError

class AnalysisStep(Generic[M, T, R, C], ABC):
    @abstractmethod
    def analyze(
        self, 
        # model: M,
        data: T, 
        results: R # evaluation results
    ) -> C:
        raise NotImplementedError

class Pipeline(Generic[P], ABC):
    @abstractmethod
    def run(self) -> P:
        raise NotImplementedError