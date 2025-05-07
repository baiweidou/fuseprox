from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 16})

def get_tensorborad_inf(path):
    event_acc = EventAccumulator(path)
    event_acc.Reload()

    draw_list = event_acc.Tags()['scalars']
    draw_dict = {}
    for name in draw_list:
        draw_dict[name] = event_acc.Scalars(name)
    return draw_dict

def smooth_value(values,weight):
    scalar = values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

if __name__ == '__main__':
    """
    确定当前模式
    """
    draw_label = '验证指标/Acc'
    # draw_label = '训练指标_Loss'

    tensorboard_log_file = '../exp/Ablation'
    tensorboard_log_list = os.listdir(tensorboard_log_file)
    index = 0
    for log_name in tqdm(tensorboard_log_list, total=len(tensorboard_log_list)):
        index = index + 1
        type_log_path = os.path.join(tensorboard_log_file,log_name)
        draw_dict = get_tensorborad_inf(type_log_path)
        for key, value in draw_dict.items():
            if '训练' in key:
                epoch = [0] + [e.step * 128 for e in value]
                values = [0] + [e.value for e in value]
            elif '验证' in key:
                epoch = [0] + [e.step * 64 for e in value]
                values = [0] + [e.value for e in value]
            log_dict = {'iteration':epoch,'values':values}
            key = key.replace('/','_')
            data_inf = pd.DataFrame(log_dict)
            data_inf.to_csv(os.path.join('../paper_file/log',log_name+"_"+key+'.csv'))



    #         if '训练' in key:
    #             # if 'Loss' in key:
    #             #     epoch = [0]+[e.step * 281 for e in value]
    #             #     values = [1]+[e.value for e in value]
    #             #     values = smooth_value(values, weight=0.00)
    #             #     plt.plot(epoch, values, label='No.{}'.format(str(index)))
    #             if 'acc' in key:
    #                 epoch = [0]+[e.step * 281 for e in value]
    #                 values = [0]+[e.value for e in value]
    #                 values = smooth_value(values, weight=0.00)
    #         if '验证' in key:
    #             # if 'loss' in key:
    #             #     epoch = [0]+[e.step * 281 for e in value]
    #             #     values = [2]+[e.value for e in value]
    #             #     values = smooth_value(values, weight=0.00)
    #             if 'Acc' in key:
    #                 epoch = [0]+[e.step * 281 for e in value]
    #                 values = [0]+[e.value for e in value]
    #                 values = smooth_value(values, weight=0.1)
    #                 plt.plot(epoch, values, label='No.{}'.format(str(index)))
    #
    # key = 'Acc'
    # if 'loss' in key or 'Loss' in key :
    #     save_name = key.replace('/','_')
    #     plt.xlabel('Iterations', fontsize=16)
    #     plt.ylabel('Loss value', fontsize=16)
    #     plt.legend(loc='upper right', ncol=2)
    #     plt.xlim(0, 56300)
    #     # plt.ylim(0.18, 0.40)
    #     plt.title('The loss curves')
    #
    #     # plt.savefig(os.path.join('Data/imgs','loss_hyp_local.png'), dpi=400)
    #     plt.show()
    # if 'Acc' in key or 'acc' in key :
    #     save_name = key.replace('/','_')
    #     plt.xlabel('Iterations', fontsize=16)
    #     plt.ylabel('Accuracy value', fontsize=16)
    #     plt.legend(loc='lower right', ncol=2)
    #     # plt.xlim(30000, 56300)
    #     # plt.ylim(0.90, .975)
    #     plt.xlim(0, 56300)
    #     plt.ylim(0, 1.0)
    #     plt.title('The accuracy curves')
    #     # plt.savefig(os.path.join('Data/imgs','acc_hyp_local.png'), dpi=400)
    #     plt.show()

