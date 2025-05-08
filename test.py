import argparse
import os
from sklearn.metrics import auc
import torch.nn as nn
import torch
import cv2

from common.logger import AverageMeter
from common.evaluation import Evaluator
from common.loss import myLoss
from common import config
from common import utils
from data.dataset import CSDataset
from models import create_model
import PIL.Image as Image
import csv
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description='JTFN for Curvilinear Structure Segmentation')
    parser.add_argument('--config', type=str, default='/config/CHUAC.yaml', help='Model config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def create_csv(path, csv_head):
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        # csv_head = ["good","bad"]
        csv_write.writerow(csv_head)

def write_csv(path, data_row):
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        # data_row = ["1","2"]
        csv_write.writerow(data_row)

def main():
    global args
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    global myloss
    myloss = myLoss()
    model = create_model(args)

    print("=> creating model ...")
    print("Classes: {}".format(args.classes))

    # Device setup
    print('# available GPUs: %d' % torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = model.cuda()
        model = nn.DataParallel(model)
        print('Use GPU Parallel.')
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model
    print("args.weight", args.weight)

    if args.weight:
        if os.path.isfile(os.path.join(args.weight,'tbd/',args.architecture,'best_model_f1.pt')):
            print("=> loading weight '{}'".format(os.path.join(args.weight,'tbd/',args.architecture,'best_model_f1.pt')))
            checkpoint = torch.load(os.path.join(args.weight,'tbd/',args.architecture,'best_model_f1.pt'))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded weight '{}'".format(os.path.join(args.weight,'tbd/',args.architecture,'best_model_f1.pt')))
        else:
            print("=> no weight found at '{}'".format(args.weight))
    else:
        raise RuntimeError("Please support weight.")

    Evaluator.initialize()

    # Dataset initialization
    CSDataset.initialize(datapath=args.datapath)
    dataclass = CSDataset.datasets[args.benchmark](args.benchmark,
                                              datapath=args.datapath,
                                              split='test',
                                              img_mode='same',
                                              img_size=args.test_size)
    L = dataclass.load_metadata()
    print(L)
    dataloader_val = CSDataset.build_dataloader(args.benchmark,
                                                args.batch_size_val,
                                                args.nworker,
                                                'test',
                                                'same',
                                                args.test_size)
    print("dataloader_val",len(dataloader_val))
    best_th = 0
    best_f1, best_pr, best_r, best_quality, best_cor, best_com = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        val_loss_dict, val_auc_, val_acc, val_f1, val_pr, val_r, val_spe, \
        val_quality, val_cor, val_com, val_cldsc, val_hd95, val_VCA = evaluate_threshold(model, dataloader_val, L, threshold=0.5)

    # show in terminal
    print('auc_: {:.4f} F1: {:.4f}  Acc: {:.4f} Recall: {:.4f} Spe: {:.4f} Precision: {:.4f} '.format(val_auc_, val_f1, val_acc, val_r, val_spe, val_pr))
    print('Quality: {:.2f} Correctness: {:.2f} Completeness: {:.2f}'.format(val_quality, val_cor, val_com))
    print('Cldsc: {:.4f} HD95: {:.2f} VCA: {:.2f}'.format(val_cldsc, val_hd95, val_VCA))
    print('==================== Finished Testing ====================')

def show_weight(model, dataloader):

    path = "/coronary_dataset/save_image/weight/"

    for idx, batch in enumerate(dataloader):
        img = batch['img']
        gt = batch['anno_mask']

        weight = np.load("weight_scale.npy")

        print(gt.shape, weight.shape)

        gt_weight_1 = gt * weight[:, 0, :, :]
        gt_weight_2 = gt * weight[:, 1, :, :]
        gt_weight_3 = gt * weight[:, 2, :, :]

        gt_weight_1 = gt_weight_1[0, 0, :, :].cpu().detach().numpy()
        gt_weight_2 = gt_weight_2[0, 0, :, :].cpu().detach().numpy()
        gt_weight_3 = gt_weight_3[0, 0, :, :].cpu().detach().numpy()

        print(gt_weight_1.shape, gt_weight_2.shape, gt_weight_3.shape)

        im = Image.fromarray(gt_weight_1 * 255.).convert("L")
        im_name = path + '{}_weight_1.png'.format(str(idx))
        im.save(im_name)

        im = Image.fromarray(gt_weight_2 * 255.).convert("L")
        im_name = path + '{}_weight_2.png'.format(str(idx))
        im.save(im_name)

        im = Image.fromarray(gt_weight_3 * 255.).convert("L")
        im_name = path + '{}_weight_3.png'.format(str(idx))
        im.save(im_name)


def save_image(prob, idx, save_dir):
    N = prob.shape[0]
    image_path = '/coronary_dataset/'+save_dir+'/'+args.benchmark+'/'+args.architecture+'/'
    utils.create_dir(image_path)
    for num in range(0, N):
        image = prob[num, 0, :, :].cpu().detach().numpy()
        im = Image.fromarray(image * 255.).convert("L")
        im_name = image_path + '{}.png'.format(idx)
        im.save(im_name)

def save_hmapimage(colored_heatmap, idx, save_dir):
    image_path = '/coronary_dataset/'+save_dir+'/'+args.benchmark+'/'+args.architecture+'/'
    utils.create_dir(image_path)
    cv2.imwrite(os.path.join(image_path,'hmap_{}.png'.format(idx)), colored_heatmap)
    print(f"Save to {os.path.join(image_path,'hmap_{}.png'.format(idx))}")

def evaluate(model, dataloader):
    global args
    args = get_parser()
    global myloss
    myloss = myLoss()

    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    val_score_path = os.path.join('logs', args.logname + '.log', 'tbd/',args.architecture) + '/' + 'test_single_image_val.csv'
    csv_head = ["image_name", "f1", "pr", "recall", "quality", "cor", "com"]
    create_csv(val_score_path, csv_head)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch
        output_dict = model(batch)
        out = output_dict['output']
        pred_mask = torch.where(out >= 0.5, 1, 0)
        save_image(out, idx)

        loss_dict = myloss(output_dict, batch_dict=batch)

        auc_, acc, f1, pr, r, spe, quality, cor, com = Evaluator.classify_prediction(pred_mask.clone(), batch) 
        img_name = batch.get('img_name')
        data_row_f1score = [str(img_name), str(auc_), str(acc), str(spe),str(f1), str(pr), str(r),
                            str(quality), str(cor), str(com)]
        write_csv(val_score_path, data_row_f1score)
        average_meter.update(auc_, acc, f1, pr, r, spe, quality, cor, com, loss_dict)



    avg_loss_dict = dict()
    for key in average_meter.loss_buf.keys():
        avg_loss_dict[key] = utils.mean(average_meter.loss_buf[key])
    f1 = average_meter.compute_f1()
    pr = average_meter.compute_precision()
    r = average_meter.compute_recall()
    quality = average_meter.compute_quality()
    cor = average_meter.compute_correctness()
    com = average_meter.compute_completeness()

    return avg_loss_dict, f1, pr, r, quality, cor, com

def evaluate_threshold(model, dataloader, L=None, threshold=0.5):
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    val_score_path = os.path.join('logs', args.logname + '.log') + '/' + 'single_image_val.csv'
    csv_head = ["image_name", "auc_", "acc", "f1", "pr", "recall","spe", "quality", "cor", "com", "cldice", "hd95","VCA"]
    create_csv(val_score_path, csv_head)

    model_pr_value, model_pr, model_r, model_auc_, model_tprs, model_fprs = [], [], [], [], [], []
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch


        if args.trans:
            out = utils.process_image_in_patches_overleap(model, batch['img'], patch_size=args.img_size)

        else:
            output_dict,_ = model(batch['img'])
            out = output_dict['output'] 

        pred_mask = torch.where(out >= threshold, 1, 0)

        heatmap = (out * 255).type(torch.uint8)  
        heatmap = heatmap.squeeze().cpu().numpy()  
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        save_image(pred_mask, L[idx], save_dir="save_image")

        loss_dict = myloss(out, batch_dict=batch)

        pr_value, auc_, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95, VCA, fpr_, tpr_, recall1_, precision1_ = Evaluator.classify_prediction(out.clone(), batch)

        model_pr_value.append(pr_value)
        model_auc_.append(auc_)
        model_tprs.append(tpr_)
        model_fprs.append(fpr_)
        model_pr.append(precision1_)
        model_r.append(recall1_)

        img_name = batch.get('img_name')
        data_row_f1score = [str(img_name), str(auc_), str(acc), str(f1), str(pr), str(r), str(spe),
                            str(quality), str(cor), str(com), str(cldsc), str(hd95), str(VCA)]
        write_csv(val_score_path, data_row_f1score)
        average_meter.update(auc_, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95, VCA, loss_dict)

    def save_to_file(data, filename):
        with open(filename, 'w') as file:
            file.write(repr(data))  

    model_pr_value = [arr.tolist() for arr in model_pr_value]
    model_auc_ = [arr.tolist() for arr in model_auc_]
    model_fprs = [arr.tolist() for arr in model_fprs]
    model_tprs = [arr.tolist() for arr in model_tprs]
    model_pr = [arr.tolist() for arr in model_pr]
    model_r = [arr.tolist() for arr in model_r]
    save_to_file(model_pr_value, '/coronary_dataset/save_pre_no_b/'+'/'+args.benchmark+'/'+args.architecture+'/'+'pr_value.txt')
    save_to_file(model_auc_, '/coronary_dataset/save_pre_no_b/'+'/'+args.benchmark+'/'+args.architecture+'/'+'auc_.txt')
    save_to_file(model_fprs, '/coronary_dataset/save_pre_no_b/'+'/'+args.benchmark+'/'+args.architecture+'/'+'fpr.txt')
    save_to_file(model_tprs, '/coronary_dataset/save_pre_no_b/'+'/'+args.benchmark+'/'+args.architecture+'/'+'tpr.txt')
    save_to_file(model_pr, '/coronary_dataset/save_pre_no_b/'+'/'+args.benchmark+'/'+args.architecture+'/'+'pr.txt')
    save_to_file(model_r, '/coronary_dataset/save_pre_no_b/'+'/'+args.benchmark+'/'+args.architecture+'/'+'r.txt')
    print("pr_value, auc_, FPR, TPR, pr, r saved to txt")

    avg_loss_dict = dict()
    for key in average_meter.loss_buf.keys():
        avg_loss_dict[key] = utils.mean(average_meter.loss_buf[key])
    auc_ = average_meter.compute_auc()
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

    return avg_loss_dict, auc_, acc, f1, pr, r, spe, quality, cor, com, cldsc, hd95,VCA


if __name__ == '__main__':
    main()