import math
import sys
import time
import torch


import torchvision.models.detection.mask_rcnn
from tqdm import tqdm


from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from .utils import reduce_dict


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    writer=None):
    model.train()
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        
        

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor,
                                                         total_iters=warmup_iters)

    for batch_idx, (images, targets) in tqdm(enumerate(data_loader),
                                             total=len(data_loader),
                                             leave=False,
                                             desc='Train batch'):
       
          
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        
        print(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if writer is not None:
            log_idx = epoch * len(data_loader) + batch_idx
            writer.add_scalar("Loss/train", losses_reduced, log_idx)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], log_idx)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, coco,  writer):
    
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
   
    model.eval()
    
#    coco = get_coco_api_from_dataset(data_loader.dataset) #is this needed for every epoch?
    
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
   
    for batch_idx, (images, targets) in tqdm(enumerate(data_loader),
                                             total=len(data_loader),
                                             leave=False, desc='Eval batch'):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        if writer is not None:
            log_idx = epoch * len(data_loader) + batch_idx
            writer.add_scalar("Time/model", model_time, log_idx)
            writer.add_scalar("Time/evaluator", evaluator_time, log_idx)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
