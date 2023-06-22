def get_metrics(gt_file, all_detections):
    ground_truth = parse_gt(gt_file)

    # The actual number of frames in the video
    nb_frames = len(all_detections)

    # Keep only the last nb_frames frames of the ground truth 
    ground_truth = ground_truth[-nb_frames:]

    for i in range(nb_frames):
        gt = ground_truth[i]
        detections = all_detections[i]
        # Build a list of all the (gt, det) pairs
        gt_det_pairs = []
        for det in detections:
            for g in gt:
                # If the class is the same, add the pair to the list
                if int(g['class_id']) == int(det['class_id']):
                    gt_det_pairs.append((g, det))
        
        gt_get_pairs_with_iou = []
        for pair in gt_det_pairs:
            pair_iou = iou(pair[0], pair[1])
            if pair_iou > 0.5:
                gt_get_pairs_with_iou.append((pair, pair_iou))
        
        # Sort the pairs by decreasing IOU
        gt_get_pairs_with_iou.sort(key=lambda x: x[1], reverse=True)

        # Count the number of true positives
        tp = 0
        while len(gt_get_pairs_with_iou) > 0:
            tp += 1
            # Remove the pair with the highest IOU from the list and remove all the pairs that have the same det
            gt_get_pairs_with_iou = [pair for pair in gt_get_pairs_with_iou if pair[0][1] != gt_get_pairs_with_iou[0][0][1]]
    
    # Compute the metrics
    nb_detections = sum([len(detections) for detections in all_detections])
    nb_gt = sum([len(gt) for gt in ground_truth])
    fp = nb_detections - tp
    fn = nb_gt - tp  
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}

def iou(bbox1, bbox2):
    # Compute the intersection
    x1 = max(bbox1['left'], bbox2['left'])
    y1 = max(bbox1['bottom'], bbox2['bottom'])
    x2 = min(bbox1['right'], bbox2['right'])
    y2 = min(bbox1['top'], bbox2['top'])
    intersection = max(0, x2 - x1) * max(0, y1 - y2)

    # Compute the union
    area1 = (bbox1['right'] - bbox1['left']) * (bbox1['bottom'] - bbox1['top'])
    area2 = (bbox2['right'] - bbox2['left']) * (bbox2['bottom'] - bbox2['top'])
    union = area1 + area2 - intersection
    
    # Compute the IOU
    return intersection / union

def parse_gt(gt_file):
    # ne pas oublier de faire 1 - y pour bottom et top
    return [
        [{'class_id': 0, 'left': 0.1, 'bottom': 0.2, 'right': 0.2, 'top': 0.1}],
    ]


if __name__ == '__main__':
    gt_file = 'gt.txt'
    detections = [
        [{'class_id': 0, 'left': 0.1, 'bottom': 0.2, 'right': 0.2, 'top': 0.1},
        {'class_id': 0, 'left': 0.2, 'bottom': 0.3, 'right': 0.3, 'top': 0.2},
        {'class_id': 0, 'left': 0.3, 'bottom': 0.4, 'right': 0.4, 'top': 0.3},
        {'class_id': 0, 'left': 0.4, 'bottom': 0.5, 'right': 0.5, 'top': 0.4},
        {'class_id': 0, 'left': 0.5, 'bottom': 0.6, 'right': 0.6, 'top': 0.5}],
    ]
    metrics = get_metrics(gt_file, detections)
    print(metrics)