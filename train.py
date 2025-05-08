import argparse
import os
import torch.optim as optim
import torch.nn as nn
import torch
torch.autograd.set_detect_anomaly(True)

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import config
from common import utils
from data.dataset import CSDataset
from models import create_model
import csv
from common.loss import myLoss

def get_parser():
    parser = argparse.ArgumentParser(description='--')
    parser.add_argument('--config', type=str, default='/config/CHUAC.yaml', help='Model config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def create_csv(path, csv_head):
    # path = "aa.csv"
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)

def write_csv(path, data_row):
    # path  = "aa.csv"
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def _save_checkpoint(epoch,model, optimizer, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    Logger.info(f'Saving a checkpoint: {filename} ...')
    torch.save(state, filename)
    print("saved!!!!!!!!!!!!!!!")

def custom_update(param, lr):
    with torch.no_grad():
        param.data -= lr * param.grad.data  
        param.data = param.data.clamp(0, 1)  

def main():
    global args
    args = get_parser() 
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    Logger.initialize(args, training=True) 

    global myloss
    myloss = myLoss()

    # Model initialization
    model = create_model(args)
    # print(model)

    Logger.info("=> creating model ...")
    Logger.info("Classes: {}".format(args.classes))
    Logger.log_params(model) #

    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    if torch.cuda.device_count() > 1: 
        model = model.cuda()
        model = nn.DataParallel(model)
        Logger.info('Use GPU Parallel.')
    elif torch.cuda.is_available(): 
        model = model.cuda()
    else:
        model = model

    other_params = [param for name, param in model.named_parameters() if 'rate' not in name]

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            [
                {'params': model.parameters(), 'lr': args.base_lr}  
            ],
            lr=args.base_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
        print('Optimizer: Adam')

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        print('Optimizer: SGD')

    if args.lr_update:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.gamma)
    else:
        scheduler = None

    if args.weight:
        if os.path.isfile(os.path.join(args.weight,args.benchmark+'_'+args.architecture+'_best_model_f1.pt')):
            Logger.info("=> loading weight '{}'".format(os.path.join(args.weight,args.benchmark+'_'+args.architecture+'_best_model_f1.pt')))
            checkpoint = torch.load(os.path.join(args.weight,args.benchmark+'_'+args.architecture+'_best_model_f1.pt'))
            model.load_state_dict(checkpoint['state_dict'])
            Logger.info("=> loaded weight '{}'".format(os.path.join(args.weight,args.benchmark+'_'+args.architecture+'_best_model_f1.pt')))

        else:
            Logger.info("=> no weight found at '{}'".format(os.path.join(args.weight,args.benchmark+'_'+args.architecture+'_best_model_f1.pt')))

    if args.resume:
        if os.path.isfile(args.resume):
            Logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            Logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            Logger.info("=> no checkpoint found at '{}'".format(args.resume))

    Evaluator.initialize()

    CSDataset.initialize(datapath=args.datapath)
    dataloader_trn = CSDataset.build_dataloader(args.benchmark,
                                                args.batch_size,
                                                args.nworker,
                                                'train',
                                                args.img_mode,
                                                args.img_size)
    dataloader_val = CSDataset.build_dataloader(args.benchmark,
                                                args.batch_size_val,
                                                args.nworker,
                                                'test',
                                                'same',
                                                args.test_size)

    best_val_auc = float('-inf')
    best_val_acc = float('-inf')
    best_val_spe = float('-inf')
    best_val_f1 = float('-inf')
    best_val_pr = float('-inf')
    best_val_r = float('-inf')
    best_val_loss = float('inf')

    val_score_path = os.path.join('logs', args.logname + '.log','tbd/',args.architecture) + '/' + 'val_retrain_f1.csv'
    csv_head = ["epoch", "total_loss","auc", "acc", "f1", "pr", "recall","spe", "quality", "cor", "com"]
    create_csv(val_score_path, csv_head) 

    for epoch in range(args.start_epoch, args.epochs):
        
        trn_loss_dict, _, _, trn_f1, _, _,_, trn_quality, _, _, _, _,_ = train(epoch, model, dataloader_trn, optimizer, scheduler)

        if args.evaluate:
            with torch.no_grad():
                val_loss_dict, val_auc, val_acc, val_f1, val_pr, val_r, val_spe, val_quality, val_cor, val_com, val_cldsc, val_hd95, val_VCA = evaluate(epoch, model, dataloader_val,optimizer) 
                data_row_f1score = [str(epoch), str(val_loss_dict['total_loss']), str(val_auc), str(val_acc), str(val_f1), str(val_pr), str(val_r), str(val_spe),
                                    str(val_quality), str(val_cor), str(val_com), str(val_cldsc), str(val_hd95), str(val_VCA)]
                write_csv(val_score_path, data_row_f1score)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    Logger.save_model_f1(model, epoch, val_f1, optimizer)

                if val_f1 >= best_val_f1 and val_r >= best_val_r:
                    best_val_f1 = val_f1
                    best_val_r = val_r
                    Logger.save_model_all(model, epoch, val_f1, val_r, optimizer)

        for key in trn_loss_dict.keys():
            Logger.tbd_writer.add_scalars(args.architecture+'/loss_train', {'trn_loss_' + str(key): trn_loss_dict[key]}, epoch)
        if args.evaluate: 
            for key in val_loss_dict.keys():
                Logger.tbd_writer.add_scalars(args.architecture+'/loss_train_val', {'trn_loss_' + str(key): trn_loss_dict[key],
                                                                      'val_loss_' + str(key): val_loss_dict[key]}, epoch)

        Logger.tbd_writer.add_scalars(args.architecture+'/f1', {'trn_f1': trn_f1, 'val_f1': val_f1}, epoch)
        Logger.tbd_writer.flush()

    print('Best val F1: ', best_val_f1) 
    print(args.architecture)
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')

def train(epoch, model, dataloader, optimizer, scheduler):
    if torch.cuda.device_count() > 1:
        model.module.train_mode() 
    else: 
        model.train()

    average_meter = AverageMeter(dataloader.dataset) 
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch 
        output_dict = model(batch['img'])
        out = output_dict['output']

        loss_dict = myloss(out, batch_dict=batch,contrast_loss=None)
        loss = loss_dict['total_loss']

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if args.lr_update:
            scheduler.step()

        pr_value, auc, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95, VCA, _,_,_,_ = Evaluator.classify_prediction(out.clone(), batch) 
        average_meter.update(auc, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95, VCA, loss_dict) 
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=1)

    average_meter.write_result('Training', epoch) 
    avg_loss_dict = dict()
    for key in average_meter.loss_buf.keys():
        avg_loss_dict[key] = utils.mean(average_meter.loss_buf[key])
    print("avg_loss_dict**********",avg_loss_dict)
    auc = average_meter.compute_auc()
    acc = average_meter.compute_acc()
    spe = average_meter.compute_spe()
    f1 = average_meter.compute_f1()
    pr = average_meter.compute_precision()
    r = average_meter.compute_recall()
    quality = average_meter.compute_quality()
    cor = average_meter.compute_correctness()
    com = average_meter.compute_completeness()
    cldsc = average_meter.compute_cldsc()
    hd95 = average_meter.compute_hd95()
    VCA = average_meter.compute_VCA()

    return avg_loss_dict, auc, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95, VCA

def evaluate(epoch, model, dataloader,optimizer):
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    average_meter = AverageMeter(dataloader.dataset) 
    for idx, batch in enumerate(dataloader): 
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch 

        if args.trans:
            out = utils.process_image_in_patches_overleap(model, batch['img'], patch_size=args.img_size)
            loss_dict = myloss(out, batch_dict=batch)
        else:
            output_dict,_ = model(batch['img'],batch['anno_mask']) #
            out = output_dict['output'] 
            loss_dict = myloss(out, batch_dict=batch)


        pr_value, auc, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95, VCA,_,_,_,_ = Evaluator.classify_prediction(out.clone(), batch)  
        average_meter.update(auc, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95, VCA, loss_dict) 
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=1) 

    average_meter.write_result('Validation', epoch)

    avg_loss_dict = dict()
    for key in average_meter.loss_buf.keys():
        avg_loss_dict[key] = utils.mean(average_meter.loss_buf[key])
    auc = average_meter.compute_auc() 
    acc = average_meter.compute_acc()
    spe = average_meter.compute_spe()
    f1 = average_meter.compute_f1()
    pr = average_meter.compute_precision() 
    r = average_meter.compute_recall()
    quality = average_meter.compute_quality()
    cor = average_meter.compute_correctness()
    com = average_meter.compute_completeness()
    cldsc = average_meter.compute_cldsc()
    hd95 = average_meter.compute_hd95()
    VCA = average_meter.compute_VCA()

    return avg_loss_dict, auc, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95, VCA

if __name__ == '__main__':
    main()
