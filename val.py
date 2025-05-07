# -*- coding:utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from utils.dataload import MultiModalDataset
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from utils.random_seed import setup_seed
from torch.utils.data import Dataset, DataLoader
"""
加载模型
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,MulticlassF1Score
from nets.model import MultiModalNet
"""
使用的数据加载
1.分两种情况进行，只需在yaml进行修改
"""

if __name__ == '__main__':
    model_list = []
    config_path = 'configs/Ablation.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    for cfg in config:
        torch.cuda.empty_cache()
        config_name = config_path.split('/')[1].replace('.yaml', '')
        setup_seed(cfg['seed'])
        model = MultiModalNet(Branch=cfg['model']['Branch']).cuda()
        experiment_name = 'MultiModalNet_'+cfg['model']['Branch']
        """
        定义损失函数、模型参数
        """
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        if cfg['optimizer'] == 'adamw':
            try:
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], betas=(cfg['momentum'], 0.99),eps=float(cfg['eps']), weight_decay=float(cfg['weight_decay']))
            except:
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg['eps']),weight_decay=float(cfg['weight_decay']))
        elif cfg['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg['eps']),weight_decay=float(cfg['weight_decay']))
        if cfg['scheduler_flag']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg['train_epoch'],eta_min=cfg['eta_min'])
        model_path = os.path.join('model_pth', config_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_weight_path = os.path.join(model_path, experiment_name + '.pth')
        """
        数据加载
        """
        train_dataset = MultiModalDataset(csv_path='../data/train.csv',data_dirs='../data/')
        val_dataset = MultiModalDataset(csv_path='../data/val.csv', data_dirs='../data/')
        train_dataloader = DataLoader(train_dataset,batch_size=cfg['train_batch_size'],num_workers=4,shuffle=True,pin_memory=True,)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], num_workers=4, shuffle=True,pin_memory=True, )
        train_numbers,val_numbers = len(train_dataset),len(val_dataset)
        """
        训练
        """
        sum_val_loss = 0.0
        val_acc = 0.0
        model.eval()
        accuracy = MulticlassAccuracy(num_classes=5).cuda()
        precision = MulticlassPrecision(num_classes=5, average='weighted').cuda()
        recall = MulticlassRecall(num_classes=5, average='weighted').cuda()
        f1_score = MulticlassF1Score(num_classes=5, average='weighted').cuda()  # Add F1 score metric
        model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout, ncols=200, position=0)
            output_map = []
            label_map = []
            for step, batch in enumerate(val_bar):
                optimizer.zero_grad()
                img, static, sequence, labels = batch[0]['imgs'].cuda(), batch[0]['static'].cuda(), batch[0]['time'].cuda(), batch[1].cuda()
                output, attn_weights = model(img, sequence, static)
                ce_loss = ce_criterion(output, labels)
                loss = ce_loss
                val_predict = torch.max(output, dim=1)[1]
                val_acc += torch.eq(val_predict, labels).sum().item()
                """
                计算训练期间的指标值
                """
                output_map.append(output)
                label_map.append(labels)
            predicted = torch.cat(output_map)
            y_true = torch.cat(label_map)
            accuracy_value, precision_value, recall_value, f1_value = val_acc / val_numbers, precision(predicted,y_true), recall(predicted, y_true), f1_score(predicted, y_true)

            model_pred_dict = {'model_name': experiment_name, 'accuracy': val_acc / val_numbers, 'precision': precision_value.item(),'recall':recall_value.item(),'f1_value':f1_value.item()}
            model_list.append(model_pred_dict)
    model_pred_csv = pd.DataFrame(model_list)
    model_pred_csv.to_csv(os.path.join('paper_file', 'Ablation.csv'), index=False)


