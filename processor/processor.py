import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from model import objectives                                        
import os.path as osp
from torch import nn
import torch.nn.functional as F
from utils.faiss_rerank import compute_jaccard_distance
import numpy as np
from sklearn.cluster import DBSCAN

# cluster images before each epoch begins
def cluster_begin_epoch(train_loader, model, args):
    device = "cuda"

    feature_size = args.embed_dim
    max_size = args.batch_size * ( len(train_loader)  )       
    image_bank = torch.zeros((max_size, feature_size)).to(device)
    index = 0

    model.to(device)
    model = model.eval()

    with torch.no_grad():
        for n_iter, batch in enumerate(train_loader):       
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch['images'].shape[0]   
            i_feats = model(batch, flag=False)

            image_bank[index: index + batch_size] = i_feats

            index = index + batch_size

        image_bank = image_bank[:index]       
        image_rerank_dist = compute_jaccard_distance(image_bank, k1=30, k2=6, search_option=0)  

        # DBSCAN cluster
        cluster = DBSCAN(eps= 0.6, min_samples=4, metric='precomputed', n_jobs=-1)

        image_pseudo_labels = cluster.fit_predict(image_rerank_dist)    

        del image_rerank_dist
    del image_bank

    return image_pseudo_labels


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("ICPG.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "cdm_loss": AverageMeter(),
        "chm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        image_pseudo_labels = cluster_begin_epoch(train_loader, model, args)

        image_num_cluster = len(set(image_pseudo_labels)) - (1 if -1 in image_pseudo_labels else 0)
        logger.info("==> Statistics for epoch [{}]: {} image clusters".format(epoch, image_num_cluster))

        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch, True, image_pseudo_labels, n_iter, epoch)  

            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            batch_size = batch['images'].shape[0]
            
            meters['loss'].update(total_loss.item(), batch_size)
            meters['cdm_loss'].update(ret.get('cdm_loss', 0), batch_size)
            meters['chm_loss'].update(ret.get('chm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("ICPG.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
