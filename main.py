import os

import os
import numpy as np
from tqdm import tqdm

from cityscapesloader import cityscapes
from modeling.deeplab import *

from torch.utils.data import DataLoader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

# define parameter
class parameter():
    def __init__(self):
        self.epochs=40
        self.batch_size = 8
        self.lr = 0.01
        self.checkname = 'deeplab-resnet'
        self.works = 4
        self.backbone='resnet'
        self.loss_type='ce'
        self.out_stride = 16
        self.lr_scheduler = 'poly'
        self.base_size = 513
        self.crop_size = 513
        self.sync_bn=True
        self.freeze_bn=False
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.nesterov = False
        self.dataset = 'cityscapes'

para=parameter()

def dataloader(para):
    data_dir = os.path.join(os.getcwd(), 'data', 'cityscapes')
    base_size=para.base_size
    crop_size=para.crop_size
    train_set = cityscapes.CityscapesSegmentation(base_size,crop_size, data_dir, split='train')
    val_set = cityscapes.CityscapesSegmentation(base_size,crop_size, data_dir, split='val')
    test_set = cityscapes.CityscapesSegmentation(base_size,crop_size, data_dir, split='test')
    num_class = train_set.NUM_CLASSES

    kwargs = {'num_workers': para.works, 'pin_memory': True}
    train_loader = DataLoader(train_set, batch_size=para.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=para.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=para.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, num_class

class buildModel(object):
    def __init__(self, para):
        self.args = para

        # Define Saver
        self.saver = Saver(para)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        self.train_loader, self.val_loader, self.test_loader, self.nclass = dataloader(para)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=para.backbone,
                        output_stride=para.out_stride,
                        sync_bn=para.sync_bn,
                        freeze_bn=para.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': para.lr},
                        {'params': model.get_10x_lr_params(), 'lr': para.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=para.momentum,
                                    weight_decay=para.weight_decay, nesterov=para.nesterov)

        # Define Criterion

        self.criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode=para.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(para.lr_scheduler, para.lr,
                                      para.epochs, len(self.train_loader))

        self.model = torch.nn.DataParallel(self.model)
        patch_replication_callback(self.model)
        self.model = self.model.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0

# 训练模型
def train(model,epoch):
    train_loss = 0.0
    model.model.train()
    tbar = tqdm(models.train_loader)
    num_img_tr = len(models.train_loader)
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()
        model.scheduler(model.optimizer, i, epoch, model.best_pred)
        model.optimizer.zero_grad()
        output = model.model(image)
        loss = model.criterion(output, target)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        model.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        # Show 10 * 3 inference results each epoch
        if i % (num_img_tr // 10) == 0:
            global_step = i + num_img_tr * epoch
            model.summary.visualize_image(model.writer, model.args.dataset, image, target, output, global_step)

    model.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
    print('[Epoch: %d, numImages: %5d]' % (epoch, i * model.args.batch_size + image.data.shape[0]))
    print('Loss: %.3f' % train_loss)

    # save checkpoint every epoch
    is_best = False
    model.saver.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.model.module.state_dict(),
        'optimizer': model.optimizer.state_dict(),
        'best_pred': model.best_pred,
    }, is_best)

def validate(model,epoch):
    model.model.eval()
    model.evaluator.reset()
    tbar = tqdm(model.val_loader, desc='\r')
    test_loss = 0.0
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']

        image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model.model(image)
        loss = model.criterion(output, target)
        test_loss += loss.item()
        tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        model.evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = model.evaluator.Pixel_Accuracy()
    Acc_class = model.evaluator.Pixel_Accuracy_Class()
    mIoU = model.evaluator.Mean_Intersection_over_Union()
    FWIoU = model.evaluator.Frequency_Weighted_Intersection_over_Union()
    model.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
    model.writer.add_scalar('val/mIoU', mIoU, epoch)
    model.writer.add_scalar('val/Acc', Acc, epoch)
    model.writer.add_scalar('val/Acc_class', Acc_class, epoch)
    model.writer.add_scalar('val/fwIoU', FWIoU, epoch)
    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % test_loss)

    new_pred = mIoU
    if new_pred > model.best_pred:
        is_best = True
        model.best_pred = new_pred
        model.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.model.module.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'best_pred': model.best_pred,
        }, is_best)


if __name__ == "__main__":

    if torch.cuda.is_available()==False:
        print("error,cannot use gpu")
        exit(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("use the device to run the network: " + device.type)

    torch.manual_seed(1)
    print('total epochs is: ',para.epochs)

    models=buildModel(para)
    for epo in range(0,para.epochs):
        print(epo, "--------->")
        train(models,epo)

        validate(models,epo)

    print('training over')
    exit(0)