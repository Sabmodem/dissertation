from core.steps import DataStep, ModelStep, EvalStep, AnalysisStep
from core.pipelines import *
from pandas import DataFrame
from dataclasses import dataclass, field
from typing import Any, Dict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

import logging

# DTO
@dataclass
class XBoostDataDTO:
    x_train: Any
    x_test: Any
    x_val: Any

    y_train: Any
    y_test: Any
    y_val: Any

@dataclass
class XBoostEvaluationResultDTO:
    y_val_pred: Any
    y_test_pred: Any

@dataclass
class XBoostAnalysisStepResult:
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float

@dataclass
class XBoostAnalysisStepResultDTO:
    validation_result: XBoostAnalysisStepResult
    test_result: XBoostAnalysisStepResult

# CONFIG
@dataclass
class XBoostDataStepConfig:
    train_temp_split_config: Dict[str, Any] = field(default_factory=dict)
    validation_test_split_config: Dict[str, Any] = field(default_factory=dict)
    vectorizer_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class XBoostModelStepConfig:
    model_config: Dict[str, Any] = field(default_factory=dict)
    train_config: Dict[str, Any] = field(default_factory=dict)
    model_filepath: str = field(default_factory=str)

class XBoostDataStep(DataStep[XBoostDataDTO]):
    def __init__(
            self, 
            data: DataFrame,
            config: XBoostDataStepConfig = XBoostDataStepConfig(),
            logger: logging.Logger = logging.getLogger(__name__)
        ):
        super().__init__()
        self._data = data
        self._config = config
        self._logger = logger
        self._result = None

    def prepare(self) -> XBoostDataDTO:
        # if self._result is not None:
        #     return self._result
        train, temp = train_test_split(self._data, **self._config.train_temp_split_config)
        val, test = train_test_split(temp, **self._config.validation_test_split_config)
        # Vectorize the text data using TF-IDF
        tfidf = TfidfVectorizer(**self._config.vectorizer_config)
        X_train = tfidf.fit_transform(train['Fillings'])
        X_val = tfidf.transform(val['Fillings'])
        X_test = tfidf.transform(test['Fillings'])
        # Convert labels to numerical format
        y_train = train['Fraud'].map({'no': 0, 'yes': 1}).values
        y_val = val['Fraud'].map({'no': 0, 'yes': 1}).values
        y_test = test['Fraud'].map({'no': 0, 'yes': 1}).values
        self._result = XBoostDataDTO(
            x_train=X_train,
            x_test=X_test,
            x_val=X_val,

            y_train=y_train,
            y_test=y_test,
            y_val=y_val
        )
        self._logger.info('XBoostDataStep', extra={'result': self._result, 'config': self._config})
        # self._logger.info('XBoostDataStep', extra={'config': self._config.__dict__})
        return self._result

class XBoostModelStep(ModelStep[xgb.XGBClassifier, XBoostDataDTO]):
    def __init__(
            self, 
            config: XBoostModelStepConfig = XBoostModelStepConfig(),
            logger: logging.Logger = logging.getLogger(__name__)
        ):
        super().__init__()
        self._config = config
        self._model = xgb.XGBClassifier(**self._config.model_config)
        self._logger = logger

    def train(self, data: XBoostDataDTO) -> xgb.XGBClassifier:
        eval_set = [(data.x_val, data.y_val)]
        self._model.fit(data.x_train, data.y_train, eval_set=eval_set)
        self._logger.info('XBoostModelStep', extra={'result': self._model, 'config': self._config})
        # self._logger.info('XBoostModelStep', extra={'config': self._config.__dict__})
        return self._model

    def save(self):
        if (self._config.model_filepath is None):
            self._logger.info("Model filepath is not set, will not save")
            return
        self._model.save_model(self._config.model_filepath)
    
    def load(self) -> xgb.XGBClassifier:
        self._model = xgb.XGBClassifier()
        self._model.load_model(self._config.model_filepath)
        return self._model
    
class XBoostEvaluateStep(EvalStep[xgb.XGBClassifier, XBoostDataDTO, XBoostEvaluationResultDTO]):
    def __init__(self, logger: logging.Logger = logging.getLogger(__name__)):
        super().__init__()
        self._logger = logger

    def evaluate(self, model: xgb.XGBClassifier, data: XBoostDataDTO) -> XBoostEvaluationResultDTO:
        y_val_pred = model.predict(data.x_val)
        y_test_pred = model.predict(data.x_test)
        result =  XBoostEvaluationResultDTO(
            y_val_pred=y_val_pred,
            y_test_pred=y_test_pred
        )
        self._logger.info('XBoostEvaluateStep', extra={'result': result})
        # self._logger.info('XBoostEvaluateStep')
        return result
    
class XBoostAnalysisStep(AnalysisStep[xgb.XGBClassifier, XBoostDataDTO, XBoostEvaluationResultDTO, XBoostAnalysisStepResult]):
    def __init__(self, logger: logging.Logger = logging.getLogger(__name__)):
        super().__init__()
        self._logger = logger

    def analyze(
            self,
            # model: xgb.XGBClassifier, 
            data: XBoostDataDTO, 
            evaluation_results: XBoostEvaluationResultDTO
        ) -> XBoostAnalysisStepResult:
        print(data, evaluation_results)
        validation_result = XBoostAnalysisStepResult(
            accuracy_score=accuracy_score(data.y_val, evaluation_results.y_val_pred),
            precision_score=precision_score(data.y_val, evaluation_results.y_val_pred),
            recall_score=recall_score(data.y_val, evaluation_results.y_val_pred),
            f1_score=f1_score(data.y_val, evaluation_results.y_val_pred)
        )
        test_result = XBoostAnalysisStepResult(
            accuracy_score=accuracy_score(data.y_test, evaluation_results.y_test_pred),
            precision_score=precision_score(data.y_test, evaluation_results.y_test_pred),
            recall_score=recall_score(data.y_test, evaluation_results.y_test_pred),
            f1_score=f1_score(data.y_test, evaluation_results.y_test_pred)
        )
        analysis_result = XBoostAnalysisStepResultDTO(validation_result=validation_result, test_result=test_result)
        # self._logger.info({'step': 'XBoostAnalysisStep', 'result': analysis_result.__dict__})
        self._logger.info('XBoostAnalysisStep', extra={'result': analysis_result})
        return analysis_result
    
class XBoostFullPipeline(FullPipeline[XBoostAnalysisStepResultDTO]):
    pass

class XboostTrainSavePipeline(TrainSavePipeline):
    pass

class XBoostLoadEvaluateAnalyzePipeline(LoadEvaluateAnalyzePipeline[XBoostAnalysisStepResultDTO]):
    pass
