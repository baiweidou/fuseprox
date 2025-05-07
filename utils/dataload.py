# -*- coding:utf-8 -*-
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, Dict

class MultiModalDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 data_dir: str): # Changed data_dirs to data_dir
        """
        多模态数据加载器 for pre-combined .npy files.
        :param csv_path:     包含索引 ('idx') 和标签 ('label') 的CSV文件路径 (e.g., 'data_processed/train.csv')
        :param data_dir:     包含大型 .npy 文件的目录路径 (e.g., 'data_processed/')
        """
        self.metadata = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.modalities = ['imgs', 'static', 'time'] # Corresponds to filenames

        # --- Load entire datasets into memory ---
        # Construct full paths to the .npy files
        imgs_path = os.path.join(self.data_dir, 'imgs_features.npy')
        static_path = os.path.join(self.data_dir, 'static_features.npy')
        time_path = os.path.join(self.data_dir, 'time_features.npy')

        print(f"Loading data for {os.path.basename(csv_path)}...")
        try:
            # Load the numpy arrays. Assumes they fit in RAM.
            # Use mmap_mode='r' if files are too large for RAM.
            self.all_imgs_data = np.load(imgs_path, allow_pickle=True)
            self.all_static_data = np.load(static_path, allow_pickle=True)
            self.all_time_data = np.load(time_path, allow_pickle=True)
            print("Data loaded successfully.")
            print(f"  Images shape: {self.all_imgs_data.shape}")
            print(f"  Static shape: {self.all_static_data.shape}")
            print(f"  Time shape: {self.all_time_data.shape}")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure the following files exist in the specified data_dir:")
            print(f" - {imgs_path}")
            print(f" - {static_path}")
            print(f" - {time_path}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            raise

        # Ensure data types are float32 for consistency with training script
        self.all_imgs_data = self.all_imgs_data.astype(np.float32)
        self.all_static_data = self.all_static_data.astype(np.float32)
        self.all_time_data = self.all_time_data.astype(np.float32)


    def __len__(self) -> int:
        # The length is determined by the number of entries in the train/val CSV
        return len(self.metadata)

    def __getitem__(self, item_index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Get metadata for the requested item (using item_index, the position in the CSV)
        meta = self.metadata.iloc[item_index]
        # Get the actual index ('idx') to use for slicing the large .npy arrays
        data_idx = meta['idx']
        label = meta['label']

        # Retrieve data for the specific index from the pre-loaded arrays
        data = {}
        try:
            data['imgs'] = torch.from_numpy(self.all_imgs_data[data_idx])
            data['static'] = torch.from_numpy(self.all_static_data[data_idx])
            data['time'] = torch.from_numpy(self.all_time_data[data_idx])
        except IndexError:
            print(f"Error: Index {data_idx} out of bounds for loaded data arrays.")
            print(f"Metadata requested item at index {item_index} which maps to data index {data_idx}.")
            print(f"Check if the CSV files ({self.metadata.shape[0]} rows) correspond correctly to the .npy files:")
            print(f"  Images shape: {self.all_imgs_data.shape}")
            print(f"  Static shape: {self.all_static_data.shape}")
            print(f"  Time shape: {self.all_time_data.shape}")
            # Return dummy data or raise an error
            # Returning dummy tensors of expected shape but zero values
            # Adjust shapes based on your actual data dimensions if needed
            dummy_img_shape = list(self.all_imgs_data.shape[1:]) # e.g., [N_PACKETS, IMG_SIZE, IMG_SIZE]
            dummy_static_shape = list(self.all_static_data.shape[1:]) # e.g., [num_static_features]
            dummy_time_shape = list(self.all_time_data.shape[1:]) # e.g., [N_PACKETS, 4]
            data['imgs'] = torch.zeros(dummy_img_shape, dtype=torch.float32)
            data['static'] = torch.zeros(dummy_static_shape, dtype=torch.float32)
            data['time'] = torch.zeros(dummy_time_shape, dtype=torch.float32)
            label = -1 # Indicate an error with the label as well
            # Or raise IndexError(f"Index {data_idx} out of bounds.")
        except Exception as e:
            print(f"Unexpected error retrieving data for index {data_idx}: {e}")
            raise


        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply transforms if any (currently none defined)
        # if self.transform:
        #     # Apply transform appropriately, might need modality-specific transforms
        #     pass

        return data, label_tensor


# 使用方法示例
if __name__ == "__main__":
    # Define data parameters pointing to the processed data directory
    config = {
        # Use val.csv for example
        'csv_path': '../../data_processed/val.csv',
        # Directory containing imgs_features.npy, static_features.npy, time_features.npy
        'data_dir': '../../data_processed/',
        'batch_size': 4, # Smaller batch for example
        'num_workers': 0 # Set to 0 for easier debugging in __main__
    }

    # Create dataset and data loader
    try:
        dataset = MultiModalDataset(
            csv_path=config['csv_path'],
            data_dir=config['data_dir'],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            shuffle=False, # Usually False for validation/testing example
            pin_memory=True,
        )

        print(f"\nTesting DataLoader with batch size {config['batch_size']}...")
        # Verify data loading
        for batch_idx, (data_dict, labels) in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            # Note: Shapes will be (batch_size, N_PACKETS, IMG_SIZE, IMG_SIZE) for imgs
            #       (batch_size, num_static_features) for static
            #       (batch_size, N_PACKETS, 4) for time
            print(f"  Image shape: {data_dict['imgs'].shape}, dtype: {data_dict['imgs'].dtype}")
            print(f"  Static features shape: {data_dict['static'].shape}, dtype: {data_dict['static'].dtype}")
            print(f"  Time series shape: {data_dict['time'].shape}, dtype: {data_dict['time'].dtype}")
            print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}, values: {labels.numpy()}")
            if batch_idx >= 2: # Show first 3 batches
                break
        print("\nDataLoader test finished.")

    except FileNotFoundError:
        print("\nError during example execution: Data files not found.")
        print("Please ensure the paths in the 'config' dictionary are correct and")
        print(f"the directory '{config['data_dir']}' contains:")
        print(" - imgs_features.npy")
        print(" - static_features.npy")
        print(" - time_features.npy")
        print(f"and the file '{config['csv_path']}' exists.")
    except Exception as e:
        print(f"\nAn error occurred during the example execution: {e}")

