# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to measure segmentation performances.
"""

import numpy as np

from scipy.optimize import linear_sum_assignment


# ### Intersection Over Union ###


def compute_iou_and_match(masks_gt, masks_pred):
    """Compute IOU between potential pairs of ground truth and predicted
    instances and match the pairs with the highest score. Background is not
    considered.

    If a ground truth instance is not matched, predicted label returned is 0
    (background).

    Parameters
    ----------
    masks_gt : np.ndarray
        Ground truth (0 is background and each instance has a unique label).
        Shape (y, x).
    masks_pred : np.ndarray
        Predicted mask (0 is background and each instance has a unique label).
        Shape (y, x).

    Returns
    -------
    labels_gt : np.ndarray
        Ground truth label with shape (nb_instances_gt,).
    labels_pred : np.ndarray
        Predicted label with shape (nb_instances_pred,).
    labels_matched : np.ndarray
        Predicted label that matches a ground truth instance with shape
        (nb_instances_gt,).
    iou : np.ndarray or None
        IOU values for each matched pair with shape (nb_instances_gt,).

    """
    # cast in integers
    masks_gt = masks_gt.astype(np.int)
    masks_pred = masks_pred.astype(np.int)

    # count overlap between ground truth instances and predicted ones
    overlap, labels_gt, labels_pred = _count_overlap(masks_gt, masks_pred)

    # case where ground truth and prediction are empty
    if len(labels_gt) == 0 and len(labels_pred) == 0:
        labels_matched = np.array([])
        iou = None

    # case where ground truth exists but prediction is empty
    elif len(labels_gt) > 0 and len(labels_pred) == 0:
        labels_matched = np.array([0 for _ in labels_gt])
        iou = np.array([0 for _ in labels_gt])

    # case where ground truth is empty but an instance is predicted
    elif len(labels_gt) == 0 and len(labels_pred) > 0:
        labels_matched = np.array([0 for _ in labels_pred])
        iou = None

    # case where ground truth exists and instances are predicted
    else:

        # compute possible IOU for each ground truth and predicted pair
        possible_iou = _compute_possible_iou(overlap, labels_gt, labels_pred)

        # match pairs with the higher IOU value
        labels_gt, labels_matched, iou = _match_instances(possible_iou, labels_gt, labels_pred)

    return labels_gt, labels_pred, labels_matched, iou


def _count_overlap(masks_gt, masks_pred):
    # flatten masks
    masks_gt_flat = masks_gt.ravel()
    masks_pred_flat = masks_pred.ravel()

    # get ground truth and predicted labels
    labels_gt = np.sort(list(set(masks_gt_flat)))
    labels_pred = np.sort(list(set(masks_pred_flat)))

    # remove 0-label if it is included (background)
    if labels_gt[0] == 0:
        labels_gt = labels_gt[1:]
    if labels_pred[0] == 0:
        labels_pred = labels_pred[1:]

    # case where ground truth or predicted labels are empty
    if len(labels_gt) == 0:
        return None, labels_gt, labels_pred
    if len(labels_pred) == 0:
        return None, labels_gt, labels_pred

    # count overlap for each pair of ground truth and predicted labels
    overlap = np.zeros((1 + labels_gt.max(), 1 + labels_pred.max()))
    for i in range(len(masks_gt_flat)):
        overlap[masks_gt_flat[i], masks_pred_flat[i]] += 1

    return overlap, labels_gt, labels_pred


def _compute_possible_iou(overlap, labels_gt, labels_pred):
    # count number of ground truth and predicted pixels
    nb_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    nb_pixels_gt = np.sum(overlap, axis=1, keepdims=True)

    # compute union
    union = nb_pixels_pred + nb_pixels_gt - overlap

    # compute possible IOU values between each pair of ground truth and
    # predicted labels
    possible_iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union != 0)

    # remove impossible ground truth and predicted pairs
    possible_iou = possible_iou[labels_gt, :]
    possible_iou = possible_iou[:, labels_pred]

    return possible_iou


def _match_instances(possible_iou, labels_gt, labels_pred):
    # match ground truth instances with predicted ones
    id_gt, id_pred = linear_sum_assignment(-possible_iou)

    # get predicted labels that match with a ground truth instance
    labels_matched_ = labels_pred[id_pred]

    # get IOU value for each match
    iou_ = possible_iou[id_gt, id_pred]

    # correct if some ground truth instances are not matched
    labels_matched = []
    iou = []
    drift = 0
    for i in range(len(labels_gt)):
        if i not in id_gt:
            labels_matched.append(0)
            iou.append(0.0)
            drift += 1

        else:
            labels_matched.append(labels_matched_[i - drift])
            iou.append(iou_[i - drift])

    # format results
    labels_matched = np.array(labels_matched).astype(np.int)
    iou = np.array(iou).astype(np.float)
    mask = iou == 0
    labels_matched[mask] = 0

    return labels_gt, labels_matched, iou


# ### Instance metrics ###


def compute_instance_metrics(masks_gt, masks_pred, labels_gt, labels_pred, labels_matched, iou):
    """Compute instance-based metrics.

    Parameters
    ----------
    masks_gt : np.ndarray
        Ground truth (0 is background and each instance has a unique label).
        Shape (y, x).
    masks_pred : np.ndarray
        Predicted mask (0 is background and each instance has a unique label).
        Shape (y, x).
    labels_gt : np.ndarray
        Ground truth label with shape (nb_instances_gt,).
    labels_pred : np.ndarray
        Predicted label with shape (nb_instances_pred,).
    labels_matched : np.ndarray
        Predicted label that matches a ground truth instance with shape
        (nb_instances_gt,).
    iou : np.ndarray or None
        IOU values for each matched pair with shape (nb_instances_gt,).

    Returns
    -------
    metrics : dict
        Dictionary with different metrics.

    """
    # initialize metrics dictionary and thresholds
    metrics = {"l_f1": [], "l_ap": [], "l_aji": []}
    thresholds = np.linspace(0.5, 0.95, num=10)

    # case where there is no ground truth instances
    if iou is None:
        for j in range(50, 100, 5):
            metrics["ap_{0}".format(j)] = np.nan
            metrics["f1_{0}".format(j)] = np.nan
            metrics["aji_{0}".format(j)] = np.nan
        metrics["mean_ap"] = np.nan
        metrics["mean_f1"] = np.nan
        metrics["mean_aji"] = np.nan
        metrics["mean_iou"] = np.nan
        metrics["mean_iou_weighted"] = np.nan
        return metrics

    for threshold in thresholds:
        # count true positive, false positive and false negative
        tp, fp, fn = _count_positive_negative(labels_gt, labels_pred, iou, threshold)

        # compute metrics
        f1 = _compute_f1(tp, fp, fn)
        ap = _compute_ap(tp, fp, fn)
        aji = _compute_aji(
            masks_gt, masks_pred, labels_gt, labels_pred, labels_matched, iou, threshold
        )

        # store metrics
        metrics["l_f1"].append(f1)
        metrics["l_ap"].append(ap)
        metrics["l_aji"].append(aji)

    # get specific metrics
    for i, j in enumerate(range(50, 100, 5)):
        metrics["ap_{0}".format(j)] = metrics["l_ap"][i]
        metrics["f1_{0}".format(j)] = metrics["l_f1"][i]
        metrics["aji_{0}".format(j)] = metrics["l_aji"][i]
    metrics["mean_ap"] = np.mean(metrics["l_ap"])
    metrics["mean_f1"] = np.mean(metrics["l_f1"])
    metrics["mean_aji"] = np.mean(metrics["l_aji"])

    # compute mean IOU and weighted mean IOU
    metrics["mean_iou"] = np.mean(iou)
    weights = []
    total_area = 0
    for label_gt in labels_gt:
        area = (masks_gt == label_gt).sum()
        weights.append(area)
        total_area += area
    weights = np.array(weights) / total_area
    metrics["mean_iou_weighted"] = (iou * weights).sum()

    return metrics


def _count_positive_negative(labels_gt, labels_pred, iou, threshold=0.5):
    # keep matched pairs with an IOU greater than the threshold
    mask = iou >= threshold

    # count true positive, false positive and false negative
    tp = mask.sum()
    fp = len(labels_pred) - tp
    fn = len(labels_gt) - tp

    return tp, fp, fn


def _compute_precision(tp, fp):
    res = tp / (tp + fp) if tp > 0 else 0.0

    return res


def _compute_recall(tp, fn):
    res = tp / (tp + fn) if tp > 0 else 0.0

    return res


def _compute_f1(tp, fp, fn):
    res = (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0.0

    return res


def _compute_ap(tp, fp, fn):
    # https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    res = tp / (tp + fp + fn) if tp > 0 else 0.0

    return res


def _compute_aji(masks_gt, masks_pred, labels_gt, labels_pred, labels_matched, iou, threshold):
    # case with ground truth instances but no predictions
    if len(labels_gt) > 0 and len(labels_pred) == 0:
        return 0

    # compute union and intersection over matched instances
    intersection = 0
    union = 0
    used_i = []
    for i, (i_gt, i_matched, iou_) in enumerate(zip(labels_gt, labels_matched, iou)):
        object_gt = masks_gt == i_gt
        if iou_ < threshold:
            union += object_gt.sum()
        else:
            object_matched = masks_pred == i_matched
            intersection_ = object_gt & object_matched
            intersection += intersection_.sum()
            union_ = object_gt | object_matched
            union += union_.sum()
            used_i.append(i_matched)

    # compute union over remaining predicted instances
    for i_pred in labels_pred:
        if i_pred not in used_i:
            object_pred = masks_pred == i_pred
            union += object_pred.sum()

    # compute aggregated jaccard index
    aji = intersection / union

    return aji


# ### Surface metrics ###


def compute_surface_metrics(masks_gt, masks_pred):
    # compute binary surfaces
    surface_gt = (masks_gt > 0).ravel()
    surface_pred = (masks_pred > 0).ravel()

    # count true positive, false positive, false negative and true negative
    tp = (surface_gt & surface_pred).sum()
    fp = (~surface_gt & surface_pred).sum()
    fn = (surface_gt & ~surface_pred).sum()
    tn = (~surface_gt & ~surface_pred).sum()

    # compute accuracy
    accuracy = (tp + tn) / surface_gt.size

    # compute f1 score
    f1 = (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0.0

    # store metrics
    metrics = {"accuracy_binary": accuracy, "f1_binary": f1}

    return metrics


# ### Metrics ###


def compute_metrics(masks_gt, masks_pred, labels_gt, labels_pred, labels_matched, iou):
    # compute instance metrics
    instance_metrics = compute_instance_metrics(
        masks_gt, masks_pred, labels_gt, labels_pred, labels_matched, iou
    )

    # compute surface metrics
    surface_metrics = compute_surface_metrics(masks_gt, masks_pred)

    # merge metrics in a dictionary
    metrics = {**instance_metrics, **surface_metrics}

    return metrics


def get_instance_metrics(masks_gt, masks_pred):
    labels_gt, labels_pred, labels_matched, iou = compute_iou_and_match(masks_gt, masks_pred)
    instance_metrics = compute_instance_metrics(
        masks_gt, masks_pred, labels_gt, labels_pred, labels_matched, iou
    )
    return instance_metrics
