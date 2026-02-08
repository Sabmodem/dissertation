import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core import Experiment, DataLoader, Evaluator, Visualizer
import config as config
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f'experiment_{datetime.now().strftime("%Y%m%d%H%M%S")}.log',
        filemode='w'
    )

def make_experiment(model_name, model, vectorizer):

    base_dir = os.path.join(os.path.curdir, 'savings', model_name)
    # plots_dir = os.path.join(base_dir, 'plots')
    # models_dir = os.path.join(base_dir, 'models')
    # summary_dir = os.path.join(base_dir, 'summary')
    plots_dir = models_dir = summary_dir = base_dir

    if config.SAVE_MODELS or config.SAVE_PLOTS or config.SAVE_SUMMARY:
        os.makedirs(base_dir, exist_ok=True)
    if config.SAVE_MODELS:
        os.makedirs(models_dir, exist_ok=True)
    if config.SAVE_PLOTS:
        os.makedirs(plots_dir, exist_ok=True)
    if config.SAVE_SUMMARY:
        os.makedirs(summary_dir, exist_ok=True)


    data_loader = DataLoader(
        vectorizer=vectorizer,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )
    evaluator = Evaluator(pos_label=config.POS_LABEL)
    visualizer = Visualizer(
        save_dir=plots_dir,
        save_plots=config.SAVE_PLOTS,
        show_plots=config.SHOW_PLOTS
    )        
    experiment = Experiment(
        name=model_name,
        model=model,
        data_loader=data_loader,
        evaluator=evaluator,
        visualizer=visualizer
    )
    results = experiment.run(
        data_file=config.DATA_FILE,
        evaluate_on_train=config.EVALUATE_ON_TRAIN,
        evaluate_on_val=config.EVALUATE_ON_VAL,
        evaluate_on_test=config.EVALUATE_ON_TEST,
        generate_plots=config.GENERATE_PLOTS
    )

    if config.SAVE_MODELS:
        experiment.save_model(filepath=f'{models_dir}/model.keras')
    if config.SAVE_SUMMARY:
        summary = experiment.get_results_summary()
        # logger = logging.getLogger(__name__)
        # logger.setLevel(logging.INFO)
        # logger.addHandler(logging.FileHandler(f'./summary.log'))
        # logger.addHandler(logging.FileHandler(f'{summary_dir}/summary.log'))
        # logger.addHandler(logging.StreamHandler(sys.stdout))
        for dataset_name, metrics in summary.items():
            logging.info(f"{dataset_name.upper()} SET:")
            for metric_name, value in metrics.items():
                logging.info(f"{metric_name:12s}: {value:.4f}")
    
    return experiment, results