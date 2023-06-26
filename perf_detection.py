import matplotlib.pyplot as plt

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
        
        gt_det_pairs_with_iou = []
        for pair in gt_det_pairs:
            plot_iou(pair[0], pair[1])
            pair_iou = iou(pair[0], pair[1])
            if pair_iou > 0.3:
                gt_det_pairs_with_iou.append((pair, pair_iou))
        
        # Sort the pairs by decreasing IOU
        gt_det_pairs_with_iou.sort(key=lambda x: x[1], reverse=True)

        # Count the number of true positives
        tp = 0
        while len(gt_det_pairs_with_iou) > 0:
            tp += 1
            # Remove the pair with the highest IOU from the list and remove all the pairs that have the same det
            gt_det_pairs_with_iou = [pair for pair in gt_det_pairs_with_iou if pair[0][1] != gt_det_pairs_with_iou[0][0][1]]
    
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
    y1 = max(bbox1['top'], bbox2['top'])
    x2 = min(bbox1['right'], bbox2['right'])
    y2 = min(bbox1['bottom'], bbox2['bottom'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute the union
    area1 = (bbox1['right'] - bbox1['left']) * (bbox1['bottom'] - bbox1['top'])
    area2 = (bbox2['right'] - bbox2['left']) * (bbox2['bottom'] - bbox2['top'])
    union = area1 + area2 - intersection

    # Compute the IOU
    return intersection / union

def parse_gt(gt_file):
    # ne pas oublier de faire 1 - y pour bottom et top
    gt_by_frame = []
    current_frame = []
    with open(gt_file, 'r') as f:
        for line in f.readlines()[1:]:
            if "Temps" in line:
                gt_by_frame.append(current_frame.copy())
                current_frame = []
            else:
                if "cube" in line.lower():
                    class_id = 0
                elif "voiture" in line.lower():
                    class_id = 2
                else:
                    class_id = 0
                coords = line.split(':')[1]
                coords = coords.split(',')
                coords = [coords.replace('(', '').replace(')', '').replace(' ', '') for coords in coords]
                coords = [float(coords) for coords in coords]
                current_frame.append({
                    'class_id': class_id,
                    'left': coords[0],
                    'top': 1 - coords[1],
                    'right': coords[2],
                    'bottom': 1 - coords[3]
                })
    return gt_by_frame

def plot_iou(bbox1, bbox2):
    plt.figure()

    #  Invert the y axis
    plt.gca().invert_yaxis()

    # Set the limits of the plot
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Plots bbox1
    plt.plot([bbox1['left'], bbox1['right']], [bbox1['top'], bbox1['top']], color='black')
    plt.plot([bbox1['left'], bbox1['right']], [bbox1['bottom'], bbox1['bottom']], color='black')
    plt.plot([bbox1['left'], bbox1['left']], [bbox1['top'], bbox1['bottom']], color='black')
    plt.plot([bbox1['right'], bbox1['right']], [bbox1['top'], bbox1['bottom']], color='black')

    # Plots bbox2
    plt.plot([bbox2['left'], bbox2['right']], [bbox2['top'], bbox2['top']], color='red')
    plt.plot([bbox2['left'], bbox2['right']], [bbox2['bottom'], bbox2['bottom']], color='red')
    plt.plot([bbox2['left'], bbox2['left']], [bbox2['top'], bbox2['bottom']], color='red')
    plt.plot([bbox2['right'], bbox2['right']], [bbox2['top'], bbox2['bottom']], color='red')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # print(iou({'left': 0.5, 'top': 0.5, 'right': 0.7, 'bottom': .8}, {'left': 0.5, 'top': 0.5, 'right': 0.75, 'bottom': 0.75}))
    plot_iou({'class_id': 0, 'left': 0.54, 'top': 0.43999999999999995, 'right': 0.59, 'bottom': 0.53}, {'class_id': 0, 'confidence': 0.8829478621482849, 'left': 0.5443516254425049, 'top': 0.2520040988922119, 'right': 0.5868278980255127, 'bottom': 0.2959097385406494})