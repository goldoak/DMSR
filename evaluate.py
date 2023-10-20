import os
import glob
import argparse
import numpy as np
import pickle as cPickle
from lib.utils import compute_mAP


parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='results/eval_real', help='result directory')
opt = parser.parse_args()


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
    evaluate()