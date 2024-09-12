import argparse
import torch

from recbole.data import create_dataset, data_preparation
from recbole.utils import (
    init_seed, get_model, get_trainer, set_color, EvaluatorType)
from recbole.data.utils import load_split_dataloaders, save_split_dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default='saved/DCNV2-Mar-26-2024_13-25-45.pth', help="model state path for test")
    parser.add_argument("--wandb", type=int, default=0, help="log to wandb")

    args, _ = parser.parse_known_args()

    if args.wandb:
        import wandb
        wandb.init(project="Recbole")
    print(set_color("Load checkpoint from: ", "blue"), args.checkpoint)
    
    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']

    # load and prepare dataloader
    init_seed(config["seed"], config["reproducibility"])
    print('Seed: ', config["seed"])

    config['show_progress'] = True

    config['eval_batch_size'] = 512
    config['metrics'] = ['NDCG', 'Recall']
    # config['metrics'] = ['Recall']
    config['topk'] = [10, 20]
    config['eval_type'] = EvaluatorType.RANKING
    config['test_neg_sample_args'] = {
        'distribution': 'uniform',
        'sample_num': 100,}
    config['valid_neg_sample_args'] = {
        'distribution': 'uniform',
        'sample_num': 100,}
    
    print('Loading data for negative sampling for ranking-based metrics.')
    try:
        print('Loading data from cache')
        config["dataloaders_save_path"] = f"cache/valid_test_data/seed_{config['seed']}/{config['dataset']}-for-{config['model']}-dataloader.pth"
        train_data, valid_data, test_data = load_split_dataloaders(config)
        print('Finished loading data from cache')
    except:
        print('Loading data from cache failed. Preparing data from scratch')
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        config["checkpoint_dir"] = f'cache/valid_test_data/seed_{config["seed"]}'
        save_split_dataloaders(config, (train_data, valid_data, test_data))
        print('Finished preparing data')

    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint['state_dict'])
    # print("neighbor_agg: ", model.neighbor_agg)
    # seed for reproducibility
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

    # get trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Original evaluation
    orig_test_result =  trainer.evaluate(test_data, load_best_model=False, model_file=None, show_progress=True)

    print(f'Eval Result: {orig_test_result}')