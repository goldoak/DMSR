import os
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from lib.sgpa import SPGANet
from lib.loss import *
from data.pose_dataset import PoseDataset
from lib.utils import setup_logger

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='CAMERA', help='CAMERA or CAMERA+Real')
parser.add_argument('--data_dir', type=str, default='./datasets/NOCS', help='data directory')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_workers', type=int, default=16, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=120, help='max number of epochs to train')
parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='checkpoints/camera', help='directory to save train results')
opt = parser.parse_args()

opt.decay_epoch = [0, 5, 10]
opt.decay_rate = [1.0, 0.6, 0.3]
opt.corr_wt = 1.0
opt.cd_wt = 5.0
opt.entropy_wt = 0.0001
opt.deform_wt = 0.01
opt.scale_wt = 0.1

def train():
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    tb_writer = tf.summary.FileWriter(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # model & loss
    estimator = SPGANet(opt.n_cat, opt.nv_prior, num_structure_points=256)
    estimator.cuda()
    estimator = nn.DataParallel(estimator)

    criterion = Loss(opt.corr_wt, opt.cd_wt, opt.entropy_wt, opt.deform_wt, opt.scale_wt)
    lowrank_criterion = LowRank_Loss()
    if opt.resume_model != '':
        estimator.load_state_dict(torch.load(opt.resume_model))

    # dataset
    train_dataset = PoseDataset(opt.dataset, 'train', opt.data_dir, opt.n_pts, opt.img_size)

    # start training
    st_time = time.time()
    train_steps = 4
    global_step = train_steps * (opt.start_epoch - 1)
    n_decays = len(opt.decay_epoch)

    assert len(opt.decay_rate) == n_decays
    for i in range(n_decays):
        if opt.start_epoch > opt.decay_epoch[i]:
            decay_count = i
    train_size = train_steps * opt.batch_size
    indices = []
    page_start = -train_size
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))

        # create optimizer and adjust learning rate if needed
        if decay_count < len(opt.decay_rate):
            if epoch > opt.decay_epoch[decay_count]:
                current_lr = opt.lr * opt.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
                decay_count += 1
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if opt.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len+real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // (4 * real_len) + 1
                data_list = random.sample(camera_indices, 3*n_repeat*real_len) + real_indices*n_repeat
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start+train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                                                       num_workers=opt.num_workers, pin_memory=True)
        estimator.train()
        for i, data in tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader)):
            pred_depth, rgb, choose, cat_id, model, prior, sRT, nocs, gt_scale_offset, points = data
            pred_depth = pred_depth.cuda()
            rgb = rgb.cuda()
            choose = choose.cuda()
            cat_id = cat_id.cuda()
            model = model.cuda()
            prior = prior.cuda()
            sRT = sRT.cuda()
            nocs = nocs.cuda()
            gt_scale_offset = gt_scale_offset.cuda()
            points = points.cuda()
            structure_points, assign_mat, deltas, scale_offset = estimator(pred_depth, rgb, choose, cat_id, prior, points)

            loss1, corr_loss, cd_loss, entropy_loss, deform_loss, scale_loss = criterion(assign_mat, deltas, prior,
                                                                                         nocs, model,
                                                                                         scale_offset, gt_scale_offset)
            loss2 = lowrank_criterion(structure_points, points)
            
            loss = 1.0 * loss1 + 1.0 * loss2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            # write results to tensorboard
            summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate', simple_value=current_lr),
                                        tf.Summary.Value(tag='train_loss', simple_value=loss),
                                        tf.Summary.Value(tag='corr_loss', simple_value=corr_loss),
                                        tf.Summary.Value(tag='cd_loss', simple_value=cd_loss),
                                        tf.Summary.Value(tag='entropy_loss', simple_value=entropy_loss),
                                        tf.Summary.Value(tag='deform_loss', simple_value=deform_loss),
                                        tf.Summary.Value(tag='lowrank_loss', simple_value=loss2),
                                        tf.Summary.Value(tag='scale_loss', simple_value=scale_loss)])
            tb_writer.add_summary(summary, global_step)
            if i % 10 == 0:
                logger.info('Batch {0} Loss:{1:f}, corr_loss:{2:f}, cd_loss:{3:f}, entropy_loss:{4:f}, deform_loss:{5:f}, lowrank_loss:{6:f}, scale_loss:{7:f}'.format(
                    i, loss.item(), corr_loss.item(), cd_loss.item(), entropy_loss.item(), deform_loss.item(), loss2.item(), scale_loss.item()))

        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))
        
        # save model after each epoch
        torch.save(estimator.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.result_dir, epoch))

if __name__ == '__main__':
    train()
