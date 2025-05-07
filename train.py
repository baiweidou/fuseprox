# -*- coding:utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import transforms, datasets # Not used directly here
# utils.dataload needs to be the modified version above
from utils.dataload import MultiModalDataset
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from utils.random_seed import setup_seed
# from torch.utils.data import Dataset # Imported in dataload
from torch.utils.data import DataLoader
import numpy as np
# import matplotlib.pyplot as plt # Not used in training loop
import argparse
"""
加载模型
"""
from nets.model import MultiModalNet
"""
使用的数据加载
1.分两种情况进行，只需在yaml进行修改 (Note: Paths are now hardcoded below, adapt if using YAML for paths)
"""

if __name__ == '__main__':
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description="训练多模态融合模型")
    parser.add_argument("--config", type=str, default="configs/Innovation.yaml", # Changed default
                        help="配置文件路径")
    args = parser.parse_args()

    config_path = args.config
    print(f"使用配置文件: {config_path}")

    # 修改文件打开方式，指定UTF-8编码
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            # Assuming the YAML contains a list of configurations
            configs = yaml.safe_load(file)
            if not isinstance(configs, list):
                # If it's a single config, wrap it in a list
                configs = [configs]
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing config file {config_path}: {e}")
        sys.exit(1)

    for cfg in configs:
        torch.cuda.empty_cache()
        # Use os.path.splitext for cleaner name extraction
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        setup_seed(cfg['seed'])

        # --- Model Initialization Update ---
        # Get Transformer parameters from config
        try:
            fusion_dim = cfg['model']['fusion_dim']
            num_heads = cfg['model']['num_heads']
            num_layers = cfg['model']['num_layers']
            attn_dropout = cfg['model']['attn_dropout']
            branch_type = cfg['model']['Branch']
            num_classes = cfg.get('num_classes', 5) # Get num_classes or default to 5
        except KeyError as e:
            print(f"Error: Missing key in 'model' section of config file: {e}")
            sys.exit(1)

        model = MultiModalNet(
            num_classes=num_classes,
            Branch=branch_type,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_dropout=attn_dropout
        ).cuda()

        # --- Experiment Naming Update ---
        experiment_name = f'MultiModalNet_{branch_type}_TransformerFusion'

        """
        定义损失函数、模型参数
        """
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        # AdamW defaults are often fine, use try-except for robustness
        try:
            if cfg['optimizer'] == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], betas=(cfg.get('momentum', 0.9), 0.999), eps=float(cfg.get('eps', 1e-8)), weight_decay=float(cfg.get('weight_decay', 0.01)))
            elif cfg['optimizer'] == 'adam':
                 optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg.get('eps', 1e-8)), weight_decay=float(cfg.get('weight_decay', 0.0)))
            else:
                print(f"Unsupported optimizer: {cfg['optimizer']}. Defaulting to AdamW.")
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
        except KeyError as e:
             print(f"Error: Missing key in optimizer/scheduler config: {e}")
             sys.exit(1)

        scheduler = None # Initialize scheduler to None
        if cfg.get('scheduler_flag', False): # Use .get for optional flags
            try:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg['train_epoch'], eta_min=float(cfg.get('eta_min', 0.0)))
            except KeyError as e:
                 print(f"Error: Missing key for scheduler config: {e}")
                 sys.exit(1)

        model_path = os.path.join('model_pth', config_name)
        os.makedirs(model_path, exist_ok=True) # Ensure directory exists
        model_weight_path = os.path.join(model_path, experiment_name + '.pth')

        """
        数据加载 (Using paths from the first script's output)
        """
        # Define base directory where 'train.csv', 'val.csv', and .npy files are located
        processed_data_dir = 'data_processed' # Changed from 'data/'

        train_csv = os.path.join(processed_data_dir, 'train.csv')
        val_csv = os.path.join(processed_data_dir, 'val.csv')

        # Check if data files exist before creating datasets
        required_files = [
            train_csv, val_csv,
            os.path.join(processed_data_dir, 'imgs_features.npy'),
            os.path.join(processed_data_dir, 'static_features.npy'),
            os.path.join(processed_data_dir, 'time_features.npy')
        ]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print("Error: Required data files are missing:")
            for f in missing_files:
                print(f" - {f}")
            print("Please run the data processing script first.")
            sys.exit(1)

        try:
            # Use the modified MultiModalDataset
            train_dataset = MultiModalDataset(csv_path=train_csv, data_dir=processed_data_dir)
            val_dataset = MultiModalDataset(csv_path=val_csv, data_dir=processed_data_dir)

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg['train_batch_size'],
                num_workers=cfg.get('num_workers', 8), # Use .get for robustness
                shuffle=True,
                pin_memory=True,
                drop_last=True # Often good for training stability
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg['val_batch_size'],
                num_workers=cfg.get('num_workers', 8),
                shuffle=False, # No need to shuffle validation data
                pin_memory=True,
            )
            train_numbers, val_numbers = len(train_dataset), len(val_dataset)
            print(f"Train samples: {train_numbers}, Val samples: {val_numbers}")
            if train_numbers == 0 or val_numbers == 0:
                print("Error: Train or validation dataset is empty. Check CSV files.")
                sys.exit(1)
        except Exception as e:
            print(f"Error during DataLoader creation: {e}")
            sys.exit(1)


        """
        训练
        """
        # Ensure the SummaryWriter directory exists
        writer_dir = os.path.join("exp", config_name, experiment_name)
        os.makedirs(writer_dir, exist_ok=True)
        writer = SummaryWriter(writer_dir, flush_secs=60)

        print(f'*********************{experiment_name}开始训练*********************')
        best_acc = 0.0 # Initialize best_acc correctly

        for epoch in range(cfg['train_epoch']):
            sum_train_loss = 0.0
            train_acc = 0.0
            model.train()
            # Use len(train_dataloader) which accounts for batch size and drop_last
            train_bar = tqdm(train_dataloader, total=len(train_dataloader), file=sys.stdout, ncols=150, position=0, desc=f"Epoch {epoch+1}/{cfg['train_epoch']} [Train]")

            for step, batch in enumerate(train_bar):
                try:
                    data_dict, labels = batch
                    # Ensure data is on the correct device
                    img = data_dict['imgs'].cuda(non_blocking=True)
                    static = data_dict['static'].cuda(non_blocking=True)
                    sequence = data_dict['time'].cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                except KeyError as e:
                    print(f"\nError: Missing key in data batch: {e}. Check MultiModalDataset output.")
                    continue # Skip this batch
                except Exception as e:
                    print(f"\nError processing batch {step}: {e}")
                    continue # Skip this batch


                optimizer.zero_grad()

                # --- Model Output Handling Update ---
                output = model(img, sequence, static)

                loss = ce_criterion(output, labels)
                loss.backward()
                optimizer.step()

                # --- Accuracy Calculation ---
                train_predict = torch.max(output, dim=1)[1]
                train_acc += torch.eq(train_predict, labels).sum().item()
                sum_train_loss += loss.item()

                # Update tqdm description
                train_bar.set_postfix_str(f'Loss: {loss.item():.3f}')

            # End of Epoch Training Steps
            if cfg.get('scheduler_flag', False) and scheduler: # Check if scheduler exists
                scheduler.step()

            # Calculate average loss and accuracy for the epoch
            # Use len(train_dataset) for accuracy calculation
            avg_train_loss = sum_train_loss / len(train_dataloader)
            epoch_train_acc = train_acc / train_numbers

            print(f'\n[Train epoch {epoch + 1}] Avg Loss:{avg_train_loss:.4f} Acc:{epoch_train_acc:.4f}')
            writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
            writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch + 1)
            if scheduler:
                 writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch + 1)


            """
            验证
            """
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_dataloader, total=len(val_dataloader), file=sys.stdout, ncols=150, position=0, desc=f"Epoch {epoch+1}/{cfg['train_epoch']} [Val]")
                for step, batch in enumerate(val_bar):
                    try:
                        data_dict, labels = batch
                        img = data_dict['imgs'].cuda(non_blocking=True)
                        static = data_dict['static'].cuda(non_blocking=True)
                        sequence = data_dict['time'].cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)
                    except KeyError as e:
                        print(f"\nError: Missing key in validation data batch: {e}. Check MultiModalDataset output.")
                        continue
                    except Exception as e:
                        print(f"\nError processing validation batch {step}: {e}")
                        continue

                    # --- Model Output Handling Update ---
                    output = model(img, sequence, static)

                    loss = ce_criterion(output, labels)
                    val_predict = torch.max(output, dim=1)[1]
                    val_acc += torch.eq(val_predict, labels).sum().item()
                    val_loss += loss.item()

                    val_bar.set_postfix_str(f'Loss: {loss.item():.3f}')

            # Calculate average loss and accuracy for validation epoch
            avg_val_loss = val_loss / len(val_dataloader)
            epoch_val_acc = val_acc / val_numbers

            print(f'\n[Val epoch {epoch + 1}] Avg Loss:{avg_val_loss:.4f} Acc:{epoch_val_acc:.4f}')
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
            writer.add_scalar('Accuracy/Validation', epoch_val_acc, epoch + 1)

            # Save best model based on validation accuracy
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                torch.save(model.state_dict(), model_weight_path)
                print(f'---> Best model saved to {model_weight_path} with accuracy: {best_acc:.4f}')

        writer.close() # Close the SummaryWriter
        print(f'*********************{experiment_name}训练结束*********************')
        print(f'Best Validation Accuracy: {best_acc:.4f}')

