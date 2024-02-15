import os
import cv2
import time
import glob
import argparse
import numpy as np
from tqdm import tqdm
import pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.sgpa import SPGANet
from lib.align import ransacPnP_LM
from lib.utils import load_depth, get_bbox, draw_detections, compute_mAP


parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='val', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='./datasets/NOCS', help='data directory')
parser.add_argument('--model', type=str, default='./pretrained/camera_model.pth', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='results/camera', help='result directory')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--num_structure_points', type=int, default=256, help='number of key-points used for pose estimation')

opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

assert opt.data in ['val', 'real_test']
if opt.data == 'val':
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
    file_path = 'CAMERA/val_list.txt'
else:
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
    file_path = 'Real/test_list.txt'

K = np.eye(3)
K[0, 0] = cam_fx
K[1, 1] = cam_fy
K[0, 2] = cam_cx
K[1, 2] = cam_cy

result_dir = opt.result_dir
result_img_dir = os.path.join(result_dir, 'images')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    os.makedirs(result_img_dir)

dpt_dir = opt.data_dir.replace('NOCS', 'dpt_output')

# path for shape & scale prior
mean_shapes = np.load('assets/mean_points_emb.npy')
with open('assets/mean_scale.pkl', 'rb') as f:
    mean_scale = cPickle.load(f)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def detect():
    # resume model
    estimator = SPGANet(opt.n_cat, opt.nv_prior, num_structure_points=opt.num_structure_points, mode='test')
    estimator.cuda()
    estimator = nn.DataParallel(estimator)
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()

    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data_dir, file_path))]
    
    # frame by frame test
    t_inference = 0.0
    t_pnp = 0.0
    inst_count = 0
    img_count = 0
    t_start = time.time()
    for img_id, path in tqdm(enumerate(img_list), total=len(img_list)):
        img_path = os.path.join(opt.data_dir, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        raw_depth = load_depth(img_path)

        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join(opt.data_dir.replace('NOCS', 'mrcnn_results'), opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        num_insts = len(mrcnn_result['class_ids'])
        f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
        f_size = np.zeros((num_insts, 3), dtype=float)

        # load dpt depth predictions
        if num_insts != 0:
            pred_depth_path = os.path.join(dpt_dir, path + '_depth.pkl')
            with open(pred_depth_path, 'rb') as f:
                pred_depth_all = cPickle.load(f)
            pred_normal_path = os.path.join(dpt_dir, path + '_normal.pkl')
            with open(pred_normal_path, 'rb') as f:
                pred_normal_all = cPickle.load(f)

        # prepare frame data
        f_sketches, f_rgb, f_choose, f_catId, f_prior, f_p2d = [], [], [], [], [], []
        valid_inst = []
        for i in range(num_insts):
            cat_id = mrcnn_result['class_ids'][i] - 1
            prior = mean_shapes[cat_id]
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])
            mask = np.logical_and(mrcnn_result['masks'][:, :, i], raw_depth > 0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) < 32:
                f_sRT[i] = np.identity(4, dtype=float)
                f_size[i] = 2 * np.amax(np.abs(prior), axis=0)
                continue
            else:
                valid_inst.append(i)
            
            # process objects with valid depth observation
            if len(choose) > opt.n_pts:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:opt.n_pts] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, opt.n_pts-len(choose)), 'wrap')

            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            p2d = np.concatenate([xmap_masked, ymap_masked], axis=1)

            rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
            rgb = cv2.resize(rgb, (opt.img_size, opt.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = norm_color(rgb)
            
            pred_depth = pred_depth_all[i]
            pred_depth = (pred_depth - np.min(pred_depth)) / (np.max(pred_depth) - np.min(pred_depth))
            pred_depth = pred_depth[np.newaxis, :, :]

            pred_normal = pred_normal_all[i]
            pred_normal = pred_normal.transpose(2, 0, 1)
            pred_sketches = np.concatenate([pred_depth, pred_normal], axis=0)

            crop_w = rmax - rmin
            ratio = opt.img_size / crop_w
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio) * opt.img_size + np.floor(col_idx * ratio)).astype(np.int64)

            # concatenate instances
            f_sketches.append(pred_sketches)
            f_rgb.append(rgb)
            f_choose.append(choose)
            f_catId.append(cat_id)
            f_prior.append(prior)
            f_p2d.append(p2d)

        if len(valid_inst):
            f_sketches = torch.cuda.FloatTensor(f_sketches)
            f_rgb = torch.stack(f_rgb, dim=0).cuda()
            f_choose = torch.cuda.LongTensor(f_choose)
            f_catId = torch.cuda.LongTensor(f_catId)
            f_prior = torch.cuda.FloatTensor(f_prior)
            
            # inference
            torch.cuda.synchronize()
            t_now = time.time()
            _, assign_mat, deltas, scale_offset = estimator(f_sketches, f_rgb, f_choose, f_catId, f_prior, points=None)

            inst_shape = f_prior + deltas
            assign_mat = F.softmax(assign_mat, dim=2)
            f_coords = torch.bmm(assign_mat, inst_shape)  # bs x n_pts x 3
            torch.cuda.synchronize()
            t_inference += (time.time() - t_now)
            f_coords = f_coords.detach().cpu().numpy()
            f_catId = f_catId.cpu().numpy()
            f_insts = inst_shape.detach().cpu().numpy()
            f_scale_offset = scale_offset.detach().cpu().numpy()
            t_now = time.time()
            for i in range(len(valid_inst)):
                inst_idx = valid_inst[i]
                nocs_coords = f_coords[i]
                f_size[inst_idx] = 2 * np.amax(np.abs(f_insts[i]), axis=0)
                scale = mean_scale[f_catId[i]] + mean_scale[f_catId[i]] * f_scale_offset[i]
                _, pred_sRT, _ = ransacPnP_LM(f_p2d[i], nocs_coords * scale, K)
                pred_sRT[:3, :3] *= scale
                if pred_sRT is None:
                    pred_sRT = np.identity(4, dtype=float)
                f_sRT[inst_idx] = pred_sRT
            t_pnp += (time.time() - t_now)
            img_count += 1
            inst_count += len(valid_inst)

        # save results
        result = {}
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        result['gt_class_ids'] = gts['class_ids']
        result['gt_bboxes'] = gts['bboxes']
        result['gt_RTs'] = gts['poses']
        result['gt_scales'] = gts['size']
        result['gt_handle_visibility'] = gts['handle_visibility']

        result['pred_class_ids'] = mrcnn_result['class_ids']
        result['pred_bboxes'] = mrcnn_result['rois']
        result['pred_scores'] = mrcnn_result['scores']
        result['pred_RTs'] = f_sRT
        result['pred_scales'] = f_size

        image_short_path = '_'.join(img_path_parsing[-3:])
        save_path = os.path.join(result_dir, 'results_{}.pkl'.format(image_short_path))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)

        # draw estimation results on images
        draw_detections(raw_rgb[:, :, ::-1], result_img_dir, 'images', img_id, K, result['pred_RTs'],
                        result['pred_scales'], result['pred_class_ids'],
                        result['gt_RTs'], result['gt_scales'], result['gt_class_ids'], draw_gt=True)
    
    # write statistics
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'w')
    messages = []
    messages.append("Total images: {}".format(len(img_list)))
    messages.append("Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
        img_count, inst_count, inst_count/img_count))
    messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference/img_count))
    messages.append("PnP time: {:06f}  Average: {:06f}/image".format(t_pnp, t_pnp/img_count))
    messages.append("Total time: {:06f}".format(time.time() - t_start))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(opt.result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, opt.result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    fw = open('{0}/eval_logs.txt'.format(opt.result_dir), 'a')
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_10_idx = degree_thres_list.index(10)
    shift_10_idx = shift_thres_list.index(10)
    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('10cm: {:.1f}'.format(pose_aps[-1, -1, shift_10_idx] * 100))
    messages.append('10 degree: {:.1f}'.format(pose_aps[-1, degree_10_idx, -1] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_10_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('10cm: {:.1f}'.format(pose_acc[-1, -1, shift_10_idx] * 100))
    messages.append('10 degree: {:.1f}'.format(pose_acc[-1, degree_10_idx, -1] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_10_idx] * 100))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


if __name__ == '__main__':
    print('Detecting ...')
    detect()
    print('Evaluating ...')
    evaluate()
