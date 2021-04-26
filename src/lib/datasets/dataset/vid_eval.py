import numpy as np
import scipy.io as sio
from collections import defaultdict

BIG_NUM = 1000000

def area(box, mode="xyxy"):

    if mode == "xyxy":
        TO_REMOVE = 1
        area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
    elif mode == "xywh":
        area = box[:, 2] * box[:, 3]
    else:
        raise RuntimeError("Should not be here")

    return area


def boxlist_iou(boxlist1, boxlist2):

    area1 = area(boxlist1, mode = "xyxy")
    area2 = area(boxlist2, mode = "xyxy")

    lt = np.maximum(np.expand_dims(boxlist1[:, :2], axis=1), boxlist2[:, :2])  # [N,M,2]
    rb = np.minimum(np.expand_dims(boxlist1[:, 2:], axis=1), boxlist2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = np.clip((rb - lt + TO_REMOVE), a_min = 0, a_max=BIG_NUM)   # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def eval_proposals_vid(pred_boxlists, gt_boxlists, iou_thresh=0.5, limit=300):
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."

    gt_overlaps = []
    num_pos = 0
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        inds = np.argsort(pred_boxlist["scores"])[::-1]
        pred_boxlist['labels'] = pred_boxlist['labels'][inds]
        pred_boxlist['bbox'] = pred_boxlist['bbox'][inds]
        pred_boxlist['scores'] = pred_boxlist['scores'][inds]

        if len(pred_boxlist['labels']) > limit:
            pred_boxlist['labels'] = pred_boxlist['labels'][:limit]
            pred_boxlist['bbox'] = pred_boxlist['bbox'][:limit]
            pred_boxlist['scores'] = pred_boxlist['scores'][:limit]

        num_pos += len(gt_boxlist['labels'])

        if len(gt_boxlist['labels']) == 0:
            continue

        if len(pred_boxlist['labels']) == 0:
            continue

        overlaps = boxlist_iou(pred_boxlist['bbox'], gt_boxlist['bbox'])

        _gt_overlaps = np.zeros(gt_boxlist['bbox'].shape[0])
        for j in range(min(len(pred_boxlist['bbox']), len(gt_boxlist['bbox']))):
            max_overlaps, argmax_overlaps = np.max(overlaps, axis=0), np.argmax(overlaps, axis=0)

            gt_ovr, gt_ind = np.max(max_overlaps, axis=0), np.argmax(max_overlaps, axis=0)
            assert gt_ovr >= 0

            box_ind = argmax_overlaps[gt_ind]

            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr

            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        gt_overlaps.append(_gt_overlaps)

    gt_overlaps = np.concatenate(gt_overlaps, axis=0)
    gt_overlaps = np.sort(gt_overlaps)

    recall = (gt_overlaps >= iou_thresh).astype(float).sum() / float(num_pos)

    return {
        "recall": recall
    }

def eval_detection_vid(pred_boxlists,
                       gt_boxlists,
                       iou_thresh=0.5,
                       motion_ranges=[[0.0, 0.7], [0.7, 0.9], [0.9, 1.0]],
                       motion_specific=False,
                       use_07_metric=False):
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."

    if motion_specific:
        motion_iou_file = "./evaluation/vid_groundtruth_motion_iou.mat"
        motion_ious = sio.loadmat(motion_iou_file)
        motion_ious = np.array(
            [[motion_ious['motion_iou'][i][0][j][0] if len(motion_ious['motion_iou'][i][0][j]) != 0 else 0 \
              for j in range(len(motion_ious['motion_iou'][i][0]))] \
             for i in range(len(motion_ious['motion_iou']))])
    else:
        motion_ious = None

    motion_ap = defaultdict(dict)
    for motion_index, motion_range in enumerate(motion_ranges):
        print("Evaluating motion iou range {} - {}".format(motion_range[0], motion_range[1]))
        prec, rec = calc_detection_vid_prec_rec(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            motion_ious=motion_ious,
            iou_thresh=iou_thresh,
            motion_range=motion_range,
        )
        ap = calc_detection_vid_ap(prec, rec, use_07_metric=use_07_metric)
        motion_ap[motion_index] = {"ap": ap, "map": np.nanmean(ap)}
    return motion_ap


def calc_detection_vid_prec_rec(gt_boxlists, pred_boxlists, motion_ious,
                                iou_thresh=0.5, motion_range=[0., 1.]):
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    pred_ignore = defaultdict(list)
    if motion_ious is None:
        motion_ious = [None] * len(gt_boxlists)
        empty_weight = 0
    else:
        all_motion_iou = np.concatenate(motion_ious, axis=0)
        empty_weight = sum([(all_motion_iou[i] >= motion_range[0]) & (all_motion_iou[i] <= motion_range[1]) for i in
                            range(len(all_motion_iou))]) / float(len(all_motion_iou))
        if empty_weight == 1:
            empty_weight = 0
    for gt_boxlist, pred_boxlist, motion_iou in zip(gt_boxlists, pred_boxlists, motion_ious):
        pred_bbox = pred_boxlist['bbox']
        pred_label = pred_boxlist['labels']
        pred_score = pred_boxlist['scores']
        gt_bbox = gt_boxlist['bbox']
        gt_label = gt_boxlist['labels']
        gt_ignore = np.zeros(len(gt_bbox))

        for gt_index, gt in enumerate(gt_bbox):
            if motion_iou:
                if motion_iou[gt_index] < motion_range[0] or motion_iou[gt_index] > motion_range[1]:
                    gt_ignore[gt_index] = 1
                else:
                    gt_ignore[gt_index] = 0

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_ignore_l = gt_ignore[gt_mask_l]

            n_pos[l] += gt_bbox_l.shape[0] - sum(gt_ignore_l)
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                pred_ignore[l].extend((empty_weight,) * pred_bbox_l.shape[0])
                continue

            # VID evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(pred_bbox_l, gt_bbox_l)

            num_obj, num_gt_obj = iou.shape

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for j in range(0, num_obj):
                iou_match = iou_thresh
                iou_match_ig = -1
                iou_match_nig = -1
                arg_match = -1
                for k in range(0, num_gt_obj):
                    if (gt_ignore_l[k] == 1) & (iou[j, k] > iou_match_ig):
                        iou_match_ig = iou[j, k]
                    if (gt_ignore_l[k] == 0) & (iou[j, k] > iou_match_nig):
                        iou_match_nig = iou[j, k]
                    if selec[k] or iou[j, k] < iou_match:
                        continue
                    if iou[j, k] == iou_match:
                        if arg_match < 0 or gt_ignore_l[arg_match]:
                            arg_match = k
                    else:
                        arg_match = k
                    iou_match = iou[j, k]

                if arg_match >= 0:
                    match[l].append(1)
                    pred_ignore[l].append(gt_ignore_l[arg_match])
                    selec[arg_match] = True
                else:
                    if iou_match_nig > iou_match_ig:
                        pred_ignore[l].append(0)
                    elif iou_match_ig > iou_match_nig:
                        pred_ignore[l].append(1)
                    else:
                        pred_ignore[l].append(sum(gt_ignore_l) / float(num_gt_obj))
                    match[l].append(0)
                    # pred_ignore[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    print(n_pos)
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)
        pred_ignore_l = np.array(pred_ignore[l])

        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        pred_ignore_l = pred_ignore_l[order]

        tps = np.logical_and(match_l == 1, np.logical_not(pred_ignore_l == 1))
        fps = np.logical_and(match_l == 0, np.logical_not(pred_ignore_l == 1))
        pred_ignore_l[pred_ignore_l == 0] = 1
        fps = fps * pred_ignore_l

        tp = np.cumsum(tps)
        fp = np.cumsum(fps)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp + np.spacing(1))
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_vid_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of VID.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in VID.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
