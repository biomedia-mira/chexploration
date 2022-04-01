import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser

image_size = (224, 224)
num_classes = 3
batch_size = 150
epochs = 50
num_workers = 4
img_data_dir = '<path_to_data>/CheXpert-v1.0/'
disease_model = 'chexpert/disease/densenet-all/version_0/checkpoints/<model_checkpoint>.ckpt'


class CheXpertDataset(Dataset):
    def __init__(self, csv_file_img, image_size, augmentation=False, pseudo_rgb = True):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = img_data_dir + self.data.loc[idx, 'path_preproc']
            img_label = np.array(self.data.loc[idx, 'race_label'], dtype='int64')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label = torch.from_numpy(sample['label'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label': sample['label']}


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, image_size, pseudo_rgb, batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(self.csv_train_img, self.image_size, augmentation=True, pseudo_rgb=pseudo_rgb)
        self.val_set = CheXpertDataset(self.csv_val_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set = CheXpertDataset(self.csv_test_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)


class ResNetDisease(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.resnet34(pretrained=True)
        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.fc.in_features
        id_layer = nn.Identity(num_features)
        self.model.fc = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


class ResNetRace(pl.LightningModule):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.num_classes = num_classes
        self.model = backbone
        freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.classifier = nn.Linear(num_features, self.num_classes)
        self.model.fc = self.classifier

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        loss = F.cross_entropy(out, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


class DenseNetDisease(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=True)
        # freeze_model(self.model)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


class DenseNetRace(pl.LightningModule):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.num_classes = num_classes
        self.model = backbone
        freeze_model(self.model)
        num_features = self.model.classifier.in_features
        self.classifier = nn.Linear(num_features, self.num_classes)
        self.model.classifier = self.classifier

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        loss = F.cross_entropy(out, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test(model, data_loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            pred = torch.softmax(model(img), dim=1)
            preds.append(pred)
            targets.append(lab)

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy()


def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    # data
    data = CheXpertDataModule(csv_train_img='../datafiles/chexpert/chexpert.sample.train.csv',
                              csv_val_img='../datafiles/chexpert/chexpert.sample.val.csv',
                              csv_test_img='../datafiles/chexpert/chexpert.sample.test.csv',
                              image_size=image_size,
                              pseudo_rgb=True,
                              batch_size=batch_size,
                              num_workers=num_workers)

    # model
    pretrained = DenseNetDisease.load_from_checkpoint(disease_model, num_classes=14)

    model_type = DenseNetRace
    model = model_type(num_classes=num_classes, backbone=pretrained.model)

    # Create output directory
    out_name = 'densenet-all-split'
    out_dir = 'chexpert/race/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=epochs,
        gpus=hparams.gpus,
        logger=TensorBoardLogger('chexpert/race', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes, backbone=pretrained.model)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    model.to(device)

    cols_names = ['class_' + str(i) for i in range(0,num_classes)]

    print('VALIDATION')
    preds_val, targets_val = test(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names)
    df['target'] = targets_val
    df.to_csv(os.path.join(out_dir, 'predictions.val.csv'), index=False)

    print('TESTING')
    preds_test, targets_test = test(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names)
    df['target'] = targets_test
    df.to_csv(os.path.join(out_dir, 'predictions.test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)
