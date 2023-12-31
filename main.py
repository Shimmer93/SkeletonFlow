from argparse import ArgumentParser
import wandb
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from data.data_api import LitDataModule
from model.model_api import LitModel
from misc.utils import load_cfg, merge_args_cfg

def main(args):
    dm = LitDataModule(hparams=args)
    model = LitModel(hparams=args)

    callbacks = [
        ModelCheckpoint(
            monitor='val_metric',
            dirpath=args.model_ckpt_dir,
            filename=args.seg_model_name+'-'+args.flow_model_name+'-{epoch}-{val_metric:.4f}',
            save_top_k=1,
            save_last=True,
            mode='min'),
        TQDMProgressBar(refresh_rate=20)
    ]

    wandb_on = True if args.dev+args.test==0 else False
    if wandb_on:
        wandb_logger = WandbLogger(
            project=args.wandb_project_name,
            save_dir=args.wandb_save_dir,
            offline=args.wandb_offline,
            log_model=False,
            job_type='train')
        wandb_logger.log_hyperparams(args)

    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        logger=wandb_logger if wandb_on else None,
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator="gpu",
        sync_batchnorm=args.sync_batchnorm,
        num_nodes=args.num_nodes,
        gradient_clip_val=args.clip_grad,
        strategy=DDPStrategy(find_unused_parameters=True) if args.strategy == 'ddp' else args.strategy,
        callbacks=callbacks,
        precision=args.precision,
        benchmark=args.benchmark
    )

    if bool(args.test):
        trainer.test(model, datamodule=dm, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoint_path)
        if args.dev==0:
            trainer.test(ckpt_path="best", datamodule=dm)

    if wandb_on:
        wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/test.yaml')
    parser.add_argument('-g', "--gpus", type=str, default=None,
                        help="Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node.")
    parser.add_argument('-d', "--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument('-n', "--num_nodes", type=int, default=1)
    parser.add_argument('-w', "--num_workers", type=int, default=4)
    parser.add_argument('-b', "--batch_size", type=int, default=2048)
    parser.add_argument('-e', "--batch_size_eva", type=int, default=1000, help='batch_size for evaluation')
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument("--data_dir", type=str, default="../../data/imagenet")
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--wandb_project_name", type=str, default="fasternet")
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--wandb_save_dir', type=str, default='./')

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # please change {WANDB_API_KEY} to your personal api_key before using wandb
    os.environ["WANDB_API_KEY"] = "60b29f8aae47df8755cbd430f0179c0cd8797bf6"

    main(args)