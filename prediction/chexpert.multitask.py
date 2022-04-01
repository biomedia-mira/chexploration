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
num_classes_disease = 14
num_classes_sex = 2
num_classes_race = 3
class_weights_race = (1.0, 6.0, 11.0) # helps with balancing accuracy, very little impact on AUC
batch_size = 150
epochs = 40
num_workers = 4
img_data_dir = '<path_to_data>/CheXpert-v1.0/'


class CheXpertDataset(Dataset):
    def __init__(self, csv_file_img, image_size, augmentation=False, pseudo_rgb = True):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb

        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = img_data_dir + self.data.loc[idx, 'path_preproc']
            img_label_disease = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label_disease[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            img_label_sex = np.array(self.data.loc[idx, 'sex_label'], dtype='int64')
            img_label_race = np.array(self.data.loc[idx, 'race_label'], dtype='int64')

            sample = {'image_path': img_path, 'label_disease': img_label_disease, 'label_sex': img_label_sex, 'label_race': img_label_race}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label_disease = torch.from_numpy(sample['label_disease'])
        label_sex = torch.from_numpy(sample['label_sex'])
        label_race = torch.from_numpy(sample['label_race'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {'image': image, 'label_disease': label_disease, 'label_sex': label_sex, 'label_race': label_race}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label_disease': sample['label_disease'], 'label_sex': sample['label_sex'], 'label_race': sample['label_race']}


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


class ResNet(pl.LightningModule):
    def __init__(self, num_classes_disease, num_classes_sex, num_classes_race, class_weights_race):
        super().__init__()
        self.num_classes_disease = num_classes_disease
        self.num_classes_sex = num_classes_sex
        self.num_classes_race = num_classes_race
        self.class_weights_race = torch.FloatTensor(class_weights_race)
        self.backbone = models.resnet34(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_sex = nn.Linear(num_features, self.num_classes_sex)
        self.fc_race = nn.Linear(num_features, self.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.fc = self.fc_connect

    def forward(self, x):
        embedding = self.backbone.forward(x)
        out_disease = self.fc_disease(embedding)
        out_sex = self.fc_sex(embedding)
        out_race = self.fc_race(embedding)
        return out_disease, out_sex, out_race

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        params_sex = params_backbone + list(self.fc_sex.parameters())
        params_race = params_backbone + list(self.fc_race.parameters())
        optim_disease = torch.optim.Adam(params_disease, lr=0.001)
        optim_sex = torch.optim.Adam(params_sex, lr=0.001)
        optim_race = torch.optim.Adam(params_race, lr=0.001)
        return optim_disease, optim_sex, optim_race

    def unpack_batch(self, batch):
        return batch['image'], batch['label_disease'], batch['label_sex'], batch['label_race']

    def process_batch(self, batch):
        img, lab_disease, lab_sex, lab_race = self.unpack_batch(batch)
        out_disease, out_sex, out_race = self.forward(img)
        loss_disease = F.binary_cross_entropy(torch.sigmoid(out_disease), lab_disease)
        loss_sex = F.cross_entropy(out_sex, lab_sex)
        loss_race = F.cross_entropy(out_race, lab_race, weight=self.class_weights_race.type_as(img))
        return loss_disease, loss_sex, loss_race

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"train_loss_disease": loss_disease, "train_loss_sex": loss_sex, "train_loss_race": loss_race})
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)

        if optimizer_idx == 0:
            return loss_disease
        if optimizer_idx == 1:
            return loss_sex
        if optimizer_idx == 2:
            return loss_race

    def validation_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"val_loss_disease": loss_disease, "val_loss_sex": loss_sex, "val_loss_race": loss_race})

    def test_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"test_loss_disease": loss_disease, "test_loss_sex": loss_sex, "test_loss_race": loss_race})


class DenseNet(pl.LightningModule):
    def __init__(self, num_classes_disease, num_classes_sex, num_classes_race, class_weights_race):
        super().__init__()
        self.num_classes_disease = num_classes_disease
        self.num_classes_sex = num_classes_sex
        self.num_classes_race = num_classes_race
        self.class_weights_race = torch.FloatTensor(class_weights_race)
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_sex = nn.Linear(num_features, self.num_classes_sex)
        self.fc_race = nn.Linear(num_features, self.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.classifier = self.fc_connect

    def forward(self, x):
        embedding = self.backbone.forward(x)
        out_disease = self.fc_disease(embedding)
        out_sex = self.fc_sex(embedding)
        out_race = self.fc_race(embedding)
        return out_disease, out_sex, out_race

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        params_sex = params_backbone + list(self.fc_sex.parameters())
        params_race = params_backbone + list(self.fc_race.parameters())
        optim_disease = torch.optim.Adam(params_disease, lr=0.001)
        optim_sex = torch.optim.Adam(params_sex, lr=0.001)
        optim_race = torch.optim.Adam(params_race, lr=0.001)
        return optim_disease, optim_sex, optim_race

    def unpack_batch(self, batch):
        return batch['image'], batch['label_disease'], batch['label_sex'], batch['label_race']

    def process_batch(self, batch):
        img, lab_disease, lab_sex, lab_race = self.unpack_batch(batch)
        out_disease, out_sex, out_race = self.forward(img)
        loss_disease = F.binary_cross_entropy(torch.sigmoid(out_disease), lab_disease)
        loss_sex = F.cross_entropy(out_sex, lab_sex)
        loss_race = F.cross_entropy(out_race, lab_race, weight=self.class_weights_race.type_as(img))
        return loss_disease, loss_sex, loss_race

    # for multiple optimizers
    def training_step(self, batch, batch_idx, optimizer_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"train_loss_disease": loss_disease, "train_loss_sex": loss_sex, "train_loss_race": loss_race})
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)

        if optimizer_idx == 0:
            return loss_disease
        if optimizer_idx == 1:
            return loss_sex
        if optimizer_idx == 2:
            return loss_race

    def validation_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"val_loss_disease": loss_disease, "val_loss_sex": loss_sex, "val_loss_race": loss_race})

    def test_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"test_loss_disease": loss_disease, "test_loss_sex": loss_sex, "test_loss_race": loss_race})


def test(model, data_loader, device):
    model.eval()
    logits_disease = []
    preds_disease = []
    targets_disease = []
    logits_sex = []
    preds_sex = []
    targets_sex = []
    logits_race = []
    preds_race = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            out_disease, out_sex, out_race = model(img)

            pred_disease = torch.sigmoid(out_disease)
            pred_sex = torch.softmax(out_sex, dim=1)
            pred_race = torch.softmax(out_race, dim=1)

            logits_disease.append(out_disease)
            preds_disease.append(pred_disease)
            targets_disease.append(lab_disease)

            logits_sex.append(out_sex)
            preds_sex.append(pred_sex)
            targets_sex.append(lab_sex)

            logits_race.append(out_race)
            preds_race.append(pred_race)
            targets_race.append(lab_race)

        logits_disease = torch.cat(logits_disease, dim=0)
        preds_disease = torch.cat(preds_disease, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)

        logits_sex = torch.cat(logits_sex, dim=0)
        preds_sex = torch.cat(preds_sex, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)

        logits_race = torch.cat(logits_race, dim=0)
        preds_race = torch.cat(preds_race, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

        counts = []
        for i in range(0,num_classes_disease):
            t = targets_disease[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_sex):
            t = targets_sex == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_race):
            t = targets_race == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds_disease.cpu().numpy(), targets_disease.cpu().numpy(), logits_disease.cpu().numpy(), preds_sex.cpu().numpy(), targets_sex.cpu().numpy(), logits_sex.cpu().numpy(), preds_race.cpu().numpy(), targets_race.cpu().numpy(), logits_race.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets_disease = []
    targets_sex = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            emb = model.backbone(img)
            embeds.append(emb)
            targets_disease.append(lab_disease)
            targets_sex.append(lab_sex)
            targets_race.append(lab_race)

        embeds = torch.cat(embeds, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

    return embeds.cpu().numpy(), targets_disease.cpu().numpy(), targets_sex.cpu().numpy(), targets_race.cpu().numpy()


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
    model_type = DenseNet
    model = model_type(num_classes_disease=num_classes_disease, num_classes_sex=num_classes_sex, num_classes_race=num_classes_race, class_weights_race=class_weights_race)

    # Create output directory
    out_name = 'densenet-all'
    out_dir = 'chexpert/multitask/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease", mode='min')

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=epochs,
        gpus=hparams.gpus,
        logger=TensorBoardLogger('chexpert/multitask', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes_disease=num_classes_disease, num_classes_sex=num_classes_sex, num_classes_race=num_classes_race, class_weights_race=class_weights_race)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    model.to(device)

    cols_names_classes_disease = ['class_' + str(i) for i in range(0,num_classes_disease)]
    cols_names_logits_disease = ['logit_' + str(i) for i in range(0, num_classes_disease)]
    cols_names_targets_disease = ['target_' + str(i) for i in range(0, num_classes_disease)]

    cols_names_classes_sex = ['class_' + str(i) for i in range(0,num_classes_sex)]
    cols_names_logits_sex = ['logit_' + str(i) for i in range(0, num_classes_sex)]

    cols_names_classes_race = ['class_' + str(i) for i in range(0,num_classes_race)]
    cols_names_logits_race = ['logit_' + str(i) for i in range(0, num_classes_race)]

    print('VALIDATION')
    preds_val_disease, targets_val_disease, logits_val_disease, preds_val_sex, targets_val_sex, logits_val_sex, preds_val_race, targets_val_race, logits_val_race = test(model, data.val_dataloader(), device)
    
    df = pd.DataFrame(data=preds_val_disease, columns=cols_names_classes_disease)
    df_logits = pd.DataFrame(data=logits_val_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.val.disease.csv'), index=False)

    df = pd.DataFrame(data=preds_val_sex, columns=cols_names_classes_sex)
    df_logits = pd.DataFrame(data=logits_val_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_sex
    df.to_csv(os.path.join(out_dir, 'predictions.val.sex.csv'), index=False)

    df = pd.DataFrame(data=preds_val_race, columns=cols_names_classes_race)
    df_logits = pd.DataFrame(data=logits_val_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_race
    df.to_csv(os.path.join(out_dir, 'predictions.val.race.csv'), index=False)

    print('TESTING')
    preds_test_disease, targets_test_disease, logits_test_disease, preds_test_sex, targets_test_sex, logits_test_sex, preds_test_race, targets_test_race, logits_test_race = test(model, data.test_dataloader(), device)
    
    df = pd.DataFrame(data=preds_test_disease, columns=cols_names_classes_disease)
    df_logits = pd.DataFrame(data=logits_test_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.disease.csv'), index=False)

    df = pd.DataFrame(data=preds_test_sex, columns=cols_names_classes_sex)
    df_logits = pd.DataFrame(data=logits_test_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_sex
    df.to_csv(os.path.join(out_dir, 'predictions.test.sex.csv'), index=False)

    df = pd.DataFrame(data=preds_test_race, columns=cols_names_classes_race)
    df_logits = pd.DataFrame(data=logits_test_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_race
    df.to_csv(os.path.join(out_dir, 'predictions.test.race.csv'), index=False)

    print('EMBEDDINGS')

    embeds_val, targets_val_disease, targets_val_sex, targets_val_race = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets_disease = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_val_sex
    df['target_race'] = targets_val_race
    df.to_csv(os.path.join(out_dir, 'embeddings.val.csv'), index=False)

    embeds_test, targets_test_disease, targets_test_sex, targets_test_race = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets_disease = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_test_sex
    df['target_race'] = targets_test_race
    df.to_csv(os.path.join(out_dir, 'embeddings.test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)
