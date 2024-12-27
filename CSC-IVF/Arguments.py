# -*- coding: utf-8 -*-

class args():
    num_input_channels=1
    num_output_channels=1
    kernel_channel=64
    kernel_size=3
    ista_iters=7 #网络层数
    iter_weight_share=False
    share_decoder=False
    
    Train_data_choose='FLIR'#'FLIR' & 'NIR'& 'COCO'
    if Train_data_choose=='FLIR':
        train_data_path = '.\\Train_data_FLIR\\'
        log_interval = 12
        epochs = 60
    elif Train_data_choose=='NIR':
        train_data_path = '.\\Train_data_NIR\\'
    elif  Train_data_choose=='COCO':
        train_data_path = 'Train_COCO\\'
        log_interval = 10
        epochs = 3
    else:
        print('Wrong!')
    train_path = '.\\Train_result\\'
    device = "cuda"
    lr = 1e-2
    is_cuda = True
    img_size=128
    batch_size=16