import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader

from data.jhmdb import SubJHMDBDataset
from data.transforms import TrainAllTransforms, TrainFrameTransforms, ValAllTransforms, ValFrameTransforms

def create_dataset(hparams, mode, all_tsfm, frm_tsfm):
    if hparams.dataset_name == 'jhmdb':
        return SubJHMDBDataset(hparams.root, mode=mode, split=hparams.split, 
                               all_tsfm=all_tsfm(hparams), frm_tsfm=frm_tsfm(hparams))
    else:
        raise NotImplementedError

def collate_fn(batch):
    transposed_batch = list(zip(*batch))
    frms = torch.stack(transposed_batch[0], dim=0)
    skls = torch.stack(transposed_batch[1], dim=0)
    flow = torch.stack(transposed_batch[2], dim=0)
    return frms, skls, flow

class LitDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = create_dataset(self.hparams, 'train', TrainAllTransforms, TrainFrameTransforms)
            self.val_dataset = create_dataset(self.hparams, 'val', ValAllTransforms, ValFrameTransforms)
        else:
            self.test_dataset = create_dataset(self.hparams, 'test', ValAllTransforms, ValFrameTransforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )