from infrastructure.xboost import *
import pandas as pd
from infrastructure.helpers import create_logger
import structuring.common.config as config

data_step_config = XBoostDataStepConfig(
    train_temp_split_config=dict(test_size=0.4, random_state=62),
    validation_test_split_config=dict(test_size=0.5, random_state=62),
    vectorizer_config=dict(max_features=5000)
)

def train_and_evaluate_on_same_dataset(dataset_name):
    logger = create_logger('xboost_train_and_evaluate_on_same_dataset')
    logger.info(f'Simple experiment. Preparing a data, training a model and evaluating it, then analyzing the results. Data: {dataset_name}')
    model_step_config = XBoostModelStepConfig(
        model_config=dict(
            reg_alpha=0.1,           # L1 regularization term on weight (increase for more regularization)
            reg_lambda=1,            # L2 regularization term on weight
            max_depth=3,             # Maximum depth of a tree (increase to make model more complex)
            min_child_weight=1,      # Minimum sum of instance weight (hessian) needed in a child
            learning_rate=0.3,       # Step size shrinkage used in update to prevents overfitting
        ),
        model_filepath=f'xboost_{dataset_name}.ubj'
    )
    data = pd.read_csv(f"{config.DATA_DIR}{dataset_name}")
    data_step = XBoostDataStep(data, data_step_config, logger)
    model_step = XBoostModelStep(model_step_config, logger)
    eval_step=XBoostEvaluateStep(logger)
    analysis_step=XBoostAnalysisStep(logger)
    pipeline = XBoostFullPipeline(
        data_step=data_step,
        model_step=model_step,
        eval_step=eval_step,
        analysis_step=analysis_step
    )
    pipeline.run()

def train_and_apply_to_all_datasets(dataset_names: list):
    from dataclasses import dataclass
    @dataclass
    class DTO:
        filename: str
        data: pd.DataFrame
        data_step: XBoostDataStep

    logger = create_logger('xboost_train_and_apply_to_all_datasets')
    logger.info(f"Complex experiment. Will prepare multiple datasets then train models in it and then will apply each trained model to each prepared dataset. Data files: {', '.join(dataset_names)}")
    data_step_dtos: list[DTO] = []

    logger.info('Preparing datasets...')
    for dataset_name in dataset_names:
        data = pd.read_csv(f"{config.DATA_DIR}{dataset_name}")
        data_step = XBoostDataStep(data, data_step_config, logger)
        data_step_dtos.append(DTO(filename=dataset_name, data=data, data_step=data_step))

    logger.info('training...')
    for data_step_dto in data_step_dtos:
        logger.info(f'training on {data_step_dto.filename}')
        model_step_config = XBoostModelStepConfig(
            model_config=dict(
                reg_alpha=0.1,           # L1 regularization term on weight (increase for more regularization)
                reg_lambda=1,            # L2 regularization term on weight
                max_depth=3,             # Maximum depth of a tree (increase to make model more complex)
                min_child_weight=1,      # Minimum sum of instance weight (hessian) needed in a child
                learning_rate=0.3,       # Step size shrinkage used in update to prevents overfitting
            ),
            model_filepath=f'xboost_{data_step_dto.filename}.ubj'
        )
        model_step = XBoostModelStep(model_step_config, logger)
        pipeline = XboostTrainSavePipeline(
            data_step=data_step_dto.data_step,
            model_step=model_step
        )
        pipeline.run()

    logger.info('evaluating and analyzing...')
    for data_step_dto_first in data_step_dtos:
        for data_step_dto_second in data_step_dtos:
            logger.info(f'evaluating {data_step_dto_first.filename} on {data_step_dto_second.filename}')
            model_step_config = XBoostModelStepConfig(
                model_config=dict(
                    reg_alpha=0.1,           # L1 regularization term on weight (increase for more regularization)
                    reg_lambda=1,            # L2 regularization term on weight
                    max_depth=3,             # Maximum depth of a tree (increase to make model more complex)
                    min_child_weight=1,      # Minimum sum of instance weight (hessian) needed in a child
                    learning_rate=0.3,       # Step size shrinkage used in update to prevents overfitting
                ),
                model_filepath=f'xboost_{data_step_dto_first.filename}.ubj'
            )
            model_step = XBoostModelStep(model_step_config, logger)
            pipeline = XBoostLoadEvaluateAnalyzePipeline(
                data_step=data_step_dto_second.data_step,
                model_step=model_step,
                eval_step=XBoostEvaluateStep(logger),
                analysis_step=XBoostAnalysisStep(logger)
            )
            pipeline.run()