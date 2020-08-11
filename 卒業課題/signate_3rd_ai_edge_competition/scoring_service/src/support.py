import cv2 as cv
import numpy as np
import json
import math

def translate_color(cls):
    if cls == 0: return (0, 0, 255)
    if cls == 1: return (0, 255, 0)
    return (255, 0, 0)


def translate_name(cls):
    if cls == 0: return 'Car'
    if cls == 1: return 'Pedestrian'
    return 'ERROR'


def iou_box(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def is_intersect(box1, box2, thresh):
    if iou_box(box1, box2) > thresh:
        return True
    else:
        return False


# sort according to score as .1, .5, .7, 1.
def order_boxes_and_scores_by_score(np_boxes, np_scores, np_classes):
    if len(np_boxes) == 0:
        return np_boxes, np_scores, np_classes

    np_boxes = np.array(np_boxes)
    np_scores = np.array(np_scores)
    np_classes = np.array(np_classes)

    np_scores_indices = np_scores.argsort()
    np_boxes = np_boxes[np_scores_indices]
    np_scores = np_scores[np_scores_indices]
    np_classes = np_classes[np_scores_indices]

    return np_boxes, np_scores, np_classes


def solve_intersect_max(list_orig, list_orig_score, list_orig_class, thresh):
    list_ret = []
    list_ret_score = []
    list_ret_class = []
    for orig in range(0, len(list_orig)):
        add = True
        for ret in range(0, len(list_ret)):
            if is_intersect(list_orig[orig], list_ret[ret], thresh) == True:
                add = False
                if list_ret_score[ret] < list_orig_score[orig]:
                    list_ret[ret] = list_orig[orig]
                    list_ret_score[ret] = list_orig_score[orig]
                    list_ret_class[ret] = list_orig_class[orig]
                break
        if add == True:
            list_ret.append(list_orig[orig])
            list_ret_score.append(list_orig_score[orig])
            list_ret_class.append(list_orig_class[orig])
    return np.asarray(list_ret), np.asarray(list_ret_score), np.asarray(list_ret_class)


def solve_intersect_avg(list_orig, list_orig_score, list_orig_class, thresh):
    if len(list_orig) == 0:
        np.asarray(list_orig), np.asarray(list_orig_score), np.asarray(list_orig_class)

    list_ret = []
    list_ret_score = []
    list_ret_class = []

    for orig in range(0, len(list_orig)):
        add = True
        for ret in range(0, len(list_ret)):
            if is_intersect(list_orig[orig], list_ret[ret], thresh) == True:
                add = False
                if list_ret_score[ret] < list_orig_score[orig]:
                    # list_ret[ret]       = (3*list_ret[ret]+7*list_orig[orig])/10.0
                    w1 = pow(list_ret_score[ret], 3)
                    w2 = pow(list_orig_score[orig], 3)
                    list_ret[ret] = (w1 * list_ret[ret] + w2 * list_orig[orig]) / (w1 + w2)
                    list_ret_score[ret] = list_orig_score[orig]
                    list_ret_class[ret] = list_orig_class[orig]
                break

        if add == True:
            list_ret.append(list_orig[orig])
            list_ret_score.append(list_orig_score[orig])
            list_ret_class.append(list_orig_class[orig])

    return np.asarray(list_ret), np.asarray(list_ret_score), np.asarray(list_ret_class)


def max_iou(boxA, boxes):
    ret = 0
    index = 0
    for k in range(0, len(boxes)):
        act = iou_box(boxA, boxes[k])
        if act > ret:
            ret = act
            index = k
    return ret, index


def normalize_image_size(image, target_width, target_height, default_background_color=0, refiner_on=False):
    height, width, _ = image.shape
    height_ratio = height / target_height
    width_ratio = width / target_width
    if width_ratio > height_ratio:
        ret = __resize_by_width(image, target_width)
    else:
        ret = __resize_by_height(image, target_height)
    height, width, _ = ret.shape
    if height < target_height or width < target_width: ret = __extend_to_exact_size(ret, target_width, target_height, default_background_color, refiner_on)
    return ret


def __resize_by_width(img, target_width):
    current_height, current_width, _ = img.shape
    target_height = int(target_width / current_width * current_height)
    ret = cv.resize(img, (target_width, target_height), interpolation=cv.INTER_LINEAR)
    return ret


def __resize_by_height(img, target_height):
    current_height, current_width, _ = img.shape
    target_width = int(target_height / current_height * current_width)
    ret = cv.resize(img, (target_width, target_height), interpolation=cv.INTER_LINEAR)
    return ret


def __extend_to_exact_size(img, target_width, target_height, default_background_color, refiner_on):
    current_height, current_width, _ = img.shape
    if refiner_on:
        left_border = 0
        top_border = 0
    else:
        left_border = int((target_width - current_width) / 2)
        top_border = int((target_height - current_height) / 2)

    right_border = target_width - current_width - left_border
    bottom_border = target_height - current_height - top_border
    ret = cv.copyMakeBorder(img, top_border, bottom_border, left_border, right_border, cv.BORDER_CONSTANT, None, default_background_color)
    return ret


def filter_boxes(box, score, classs, width, height, min_size, deleted_boxes):
    dummy = []
    if len(box) == 0:
        return box, score, classs, dummy
    box[:, 0] = np.clip(box[:, 0], 0, height)
    box[:, 1] = np.clip(box[:, 1], 0, width)
    box[:, 2] = np.clip(box[:, 2], 0, height)
    box[:, 3] = np.clip(box[:, 3], 0, width)
    box = box.astype(int)
    dims = (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])
    box = box.tolist()
    score = score.tolist()
    classs = classs.tolist()
    for k in range(len(box) - 1, -1, -1):
        if dims[k] < min_size:
            del box[k]
            del score[k]
            del classs[k]

    for k in range(len(box) - 1, -1, -1):
        if classs[k] == 0:
            if score[k] < 0.4:  # TODO 0.4, epic 0.2
                #if score[k]>0.3:
                #    deleted_boxes.append(box[k])
                del box[k]
                del score[k]
                del classs[k]
        elif classs[k] == 1:
            if score[k] < 0.6:  # TODO 0.6, epic 0.3
                if score[k] > 0.55:
                    deleted_boxes.append(box[k])
                del box[k]
                del score[k]
                del classs[k]

    return np.asarray(box), np.asarray(score), np.asarray(classs), deleted_boxes


def correct_by_classifier(box, score, classs, imgcv, classifier):
    if len(box) > 0:
        imgcv_col = cv.cvtColor(imgcv, cv.COLOR_RGB2BGR)
        to_pred = []
        for k in range(0, len(box)):
            to_pred.append(normalize_image_size(imgcv_col[int(box[k][0]):int(box[k][2]), int(box[k][1]):int(box[k][3]), :], 300, 300, 0))
        pred = classifier.predict(np.asarray(to_pred), batch_size=len(to_pred))
        classs = np.argmax(pred, axis=1)
        box = box.tolist()
        score = score.tolist()
        classs = classs.tolist()
        for k in range(len(box) - 1, -1, -1):
            if classs[k] > 1:  # discard other classes than car and pedestrian
                del box[k]
                del score[k]
                del classs[k]
        return np.asarray(box), np.asarray(score), np.asarray(classs)
    return box, score, classs


# TODO: border
def correct_by_refiner(box, imgcv, refiner):
    if len(box) == 0: return box

    imgcv_col = cv.cvtColor(imgcv, cv.COLOR_RGB2BGR)
    to_pred = []
    multipliers = []
    borders = []
    border_tb = 15
    border_lr = 5
    width_im = imgcv.shape[1] - 1
    height_im = imgcv.shape[0] - 1
    for k in range(0, len(box)):
        px1 = box[k][1]
        py1 = box[k][0]
        px2 = box[k][3]
        py2 = box[k][2]

        sx1 = min(border_lr, px1)
        sx2 = min(border_lr, width_im - px2)
        sy1 = min(border_tb, py1)
        sy2 = min(border_tb, height_im - py2)
        borders.append([sy1, sx1, sy2, sx2])

        to_pred.append(normalize_image_size(imgcv_col[int(py1 - sy1):int(py2 + sy2), int(px1 - sx1):int(px2 + sx2), :], 300, 300, 0, True))

        width = (px2 + sx2) - (px1 - sx1)
        height = (py2 + sy2) - (py1 - sy1)
        multipliers.append(min(300.0 / width, 300.0 / height))

    pred = refiner.predict(np.asarray(to_pred), batch_size=len(to_pred))

    pred = np.asarray(pred)
    box = np.asarray(box)
    borders = np.asarray(borders)

    pred[..., 0] = pred[..., 0] * 300 / multipliers
    pred[..., 1] = pred[..., 1] * 300 / multipliers
    pred[..., 2] = pred[..., 2] * 300 / multipliers
    pred[..., 3] = pred[..., 3] * 300 / multipliers

    ax1 = pred[..., 1].copy()
    ax3 = pred[..., 3].copy()
    pred[..., 1] = pred[..., 0].copy()
    pred[..., 3] = pred[..., 2].copy()
    pred[..., 0] = ax1
    pred[..., 2] = ax3

    pred[..., 0] += box[..., 0] - borders[..., 0]
    pred[..., 1] += box[..., 1] - borders[..., 1]
    pred[..., 2] += box[..., 0] - borders[..., 0]
    pred[..., 3] += box[..., 1] - borders[..., 1]

    pred = pred.astype('int')

    return pred


max_boxes_tracker = 40
resolution_tracker = 224
latent_tracker = 1280


def solve_id_and_iou(pred_last, pred_act, last_ids, act_id, all_ids, box):
    #we do not have actual detection, returning
    if len(pred_act) == 0:
        return act_id, [], all_ids

    #we do not have last detections, so all ids will be new
    #todo: search in last_ids
    if len(pred_last) == 0:
        boxes_ids = np.full(len(pred_act), -1)
        for i in range(0, len(pred_act)):
            boxes_ids[i] = act_id
            act_id += 1
        return act_id, boxes_ids, all_ids

    #the ids of actual boxes. The goal is to set there paired ids and return them
    boxes_ids = np.zeros(len(pred_act), dtype=int)
    boxes_ids[:] = -1

    #compute matrix of distances for last and actual predicted vectors
    matrix_dists = np.zeros((len(pred_last), len(pred_act)))
    for i in range(0, len(pred_act)):
        for j in range(0, len(pred_last)):
            matrix_dists[j, i] = sum(pow(pred_last[j] - pred_act[i], 2.0))

    #print('matrix has shape:', matrix_dists.shape)

    for j in range(0, len(pred_last)):
        index_box = np.argmin(matrix_dists[j,:])
        index_track_box = np.argmin(matrix_dists[:, index_box])
        #print('\t searched indexes', index_box, index_track_box, j)
        #print('shape pred last, last_ids', len(pred_last), len(last_ids))

        #here we have a match
        if j == index_track_box:
            boxes_ids[index_box] = last_ids[j]
            for k in range(0, len(all_ids)):
                if all_ids[k][0] == boxes_ids[index_box]:
                    # boxes are too distant, we do not want them
                    if math.sqrt(pow(all_ids[k][6] - box[index_box][0], 2) + pow(all_ids[k][7] - box[index_box][1], 2)) > 250:
                        break

                #if all_ids[k][0] == boxes_ids[index_box] and matrix_dists[j, i] < (all_ids[k][5]+0.1)*3.0:
                    all_ids[k][3] = pred_act[index_box] #latent represent
                    all_ids[k][2] = 0 #time dead
                    all_ids[k][5] = matrix_dists[j,i] #latent distance
                    all_ids[k][6] = box[index_box][0]
                    all_ids[k][7] = box[index_box][1]
                    break

    #We set all boxes in last_ids as not used and then set as used only those with id
    for i in range (0, len(all_ids)):
        all_ids[i][1] = False
    for j in range (0, len(boxes_ids)):
        if boxes_ids[j] == -1: continue
        for i in range(0, len(all_ids)):
            if boxes_ids[j] == all_ids[i][0]:
                all_ids[i][1] = True
                break

    #now we try to search unmatched boxes in all last ids
    for i in range(0, len(boxes_ids)):
        if boxes_ids[i] != -1: continue
        min_dist = 66666
        index = -1
        #search for the min dist in latent representation
        for j in range(0, len(all_ids)):
            if all_ids[j][1] == True: continue
            dist = sum(pow(all_ids[j][3] - pred_act[i], 2.0))
            if dist < min_dist:
                min_dist = dist
                index = j


        #here we found match with last ids and realize cross double check for the min
        #print('dist: ', all_ids[index][5], min_dist)
        if index != -1:
            # double check
            min_dist2 = 66666
            index2 = -1
            for k in range(0, len(pred_act)):
                if boxes_ids[k] != -1: continue #search only within the free preds
                dist2 = sum(pow(all_ids[index][3] - pred_act[k], 2.0))
                if dist2 < min_dist2:
                    min_dist2 = dist2
                    index2 = k
            if index2 != i:
                min_dist = 6666

            #we do not want too distant boxes
            if math.sqrt(pow(all_ids[index][6] - box[i][0],2)+pow(all_ids[index][7] - box[i][1],2)) > 250:
                break


            if min_dist < 1.8:
            #if min_dist < ((all_ids[index][5])+0.1)*3.0:
                boxes_ids[i] = all_ids[index][0]
                all_ids[index][1] = True
                all_ids[index][2] = 0
                all_ids[index][3] = pred_act[i]
                all_ids[index][5] = min_dist
                all_ids[index][6] = box[i][0]
                all_ids[index][7] = box[i][1]

    #here are brand new ids, so we add into all_ids new record
    for i in range(0, len(boxes_ids)):
        if boxes_ids[i] == -1:
            boxes_ids[i] = act_id

            rec = []
            rec.append(act_id)
            rec.append(True)
            rec.append(0)
            rec.append(pred_act[i])
            rec.append(1) #some class, todo
            rec.append(0.1) #measured latent distance
            rec.append(box[i][0])  # measured latent distance
            rec.append(box[i][1])  # measured latent distance
            all_ids.append(rec)
            act_id += 1
    return act_id, boxes_ids, all_ids


def init_trackers(img, box, act_id, matcher, all_ids, classes):
    last_ids = []
    last_boxes_temp = []
    last_boxes = np.zeros((max_boxes_tracker, 224, 224, 3))

    if len(box) == 0:
        return act_id, last_boxes, np.full((max_boxes_tracker), -1), all_ids

    crops = []
    for k in range(0, len(box)):
        crops.append(normalize_image_size(img[int(box[k][1]+3):int(box[k][3]-3), int(box[k][0]+3):int(box[k][2]-3), ...], 224, 224))
    pred = matcher.predict([crops, crops, crops], batch_size=len(crops))

    for k in range(0, len(box)):
        last_boxes_temp.append(crops[k])
        last_ids.append(act_id)

        rec = []
        rec.append(act_id) #id
        rec.append(True) #is used
        rec.append(0) #time dead
        rec.append(pred[k, 0:latent_tracker]) #latent representation
        rec.append(classes[k]) #class
        rec.append(0.1) #measured latent distance
        rec.append(box[k][0]) #x position of top-left
        rec.append(box[k][1])  # y position of top-left
        all_ids.append(rec)
        act_id += 1

    for k in range(len(box), max_boxes_tracker):
        last_ids.append(-1)

    last_boxes[0:min(max_boxes_tracker, len(last_boxes_temp)), ...] = last_boxes_temp

    return act_id, last_boxes, last_ids, all_ids


def update_trackers(img, box, act_id, last_boxes, last_ids, matcher, all_ids, classs):
    #act_ids = []
    act_boxes_temp = []
    act_boxes = np.zeros((max_boxes_tracker, resolution_tracker, resolution_tracker, 3))

    #we do not have boxes, so all in all_ids are set as not used
    if len(box) == 0:
        for i in range (0, len(all_ids)):
            all_ids[i][1] = False
        return act_id, act_boxes, np.full((max_boxes_tracker), -1), all_ids

    for k in range(0, len(box)):
        crop = normalize_image_size(img[int(box[k][1]+3):int(box[k][3]-3), int(box[k][0]+3):int(box[k][2]-3), ...], resolution_tracker, resolution_tracker)
        act_boxes_temp.append(crop)
    act_boxes[0:min(max_boxes_tracker, len(act_boxes_temp)), ...] = act_boxes_temp
    pred = matcher.predict([last_boxes, act_boxes, act_boxes], batch_size=max_boxes_tracker)

    pred_last = []
    pred_act = []
    for i in range(0, max_boxes_tracker):
        pred_last.append(pred[i, 0:latent_tracker])
        pred_act.append(pred[i, latent_tracker:latent_tracker * 2])

    pred_last = np.asarray(pred_last)
    pred_act = np.asarray(pred_act)
    pred_act = pred_act[:len(box)]
    pred_last = pred_last[:len(last_ids)]

    act_id, act_ids, all_ids = solve_id_and_iou(pred_last, pred_act, last_ids, act_id, all_ids, box)

    return act_id, act_boxes, act_ids, all_ids


def solve_deleted_boxes(deleted_boxes, box, score, classs, all_ids, img, matcher, last_boxes, last_ids):
    todo = []

    if len(deleted_boxes) == 0:
        return box, score, classs, todo

    box = box.tolist()
    classs = classs.tolist()
    last_ids = last_ids.tolist()
    score = score.tolist()

    del_boxes_crops = []

    for k in range(0, len(deleted_boxes)):
        crop = normalize_image_size(img[int(deleted_boxes[k][0]):int(deleted_boxes[k][2]), int(deleted_boxes[k][1]):int(deleted_boxes[k][3]), ...], resolution_tracker, resolution_tracker)
        del_boxes_crops.append(crop)
    pred = matcher.predict([del_boxes_crops, del_boxes_crops, del_boxes_crops], batch_size=len(del_boxes_crops))

    #browse all vectors of deleted boxes
    for i in range(0, len(pred)):
        vec = pred[i][0:latent_tracker]

        min_dist = 66666
        index = -1
        for j in range(0, len(all_ids)):
            if all_ids[j][1] == True: continue #we do not want used ids
            dist = sum(pow(all_ids[j][3] - vec, 2.0))
            if dist < min_dist:
                min_dist = dist
                index = j

        if index != -1:
            min_dist2 = 66666
            index2 = -1
            for k in range(0, len(pred)):
                #todo: search only within pedestrinas
                dist2 = sum(pow(all_ids[index][3] - pred[k][0:latent_tracker], 2.0))
                if dist2 < min_dist2:
                    min_dist2 = dist2
                    index2 = k
            if index2 != i:
                min_dist = 6666

        if min_dist < 0.5:  # TODO
            #print(len(last_ids), len(last_boxes))
            #print(last_ids)

            todo.append(deleted_boxes[i])
            all_ids[index][1] = True
            all_ids[index][2] = 0
            all_ids[index][3] = vec
            print('dimensions: ', len(box), len(classs), len(last_boxes), len(last_ids)) #4,4,40,3
            if len(last_boxes) < len(last_ids):
                print('cannot append box, there is no space')
                continue
            print('adding', deleted_boxes[i])
            box.append([int(deleted_boxes[i][0]), int(deleted_boxes[i][1]), int(deleted_boxes[i][2]), int(deleted_boxes[i][3]) ])
            classs.append(1) #we add only pedestrians
            score.append(0.9) #some dummy score
            last_boxes[len(last_ids)] = del_boxes_crops[i]
            last_ids.append(all_ids[index][0])



    return np.asarray(box), np.asarray(score), np.asarray(classs), todo, last_boxes, np.asarray(last_ids)



def filter_aspect_ratio(box, score, classs, type=1, thresh=4.0):
    if len(box) == 0:
        return box, score, classs

    #height / width
    ar = (box[:, 2] - box[:, 0]) / (box[:, 3] - box[:, 1])
    box = box.tolist()
    score = score.tolist()
    classs = classs.tolist()

    for k in range(len(box) - 1, -1, -1):
        if classs[k] == type:
            if ar[k] > thresh:
                del box[k]
                del score[k]
                del classs[k]


    return np.asarray(box), np.asarray(score), np.asarray(classs)



def filter_out_ids_with_low_occurrence(json, min_required_occurence_count, verbose=False):
    """
    Removes objects with the occurrence between frames in each video lower than the specified limit

    Params:
      json
        represents json data on which the operation will be applied
      min_required_occurrence_count
        minimal number of occurrences in the video to object be kept in JSON
      verbose
        print progress comments, optional (False)

    Returns:
        Nothing.

    The changes are done directly over the passed JSON object. There is no local copy created.
    """

    if verbose: print("Filtering out id's with low occurrence")
    tmp = {}
    for movie_id in json.keys():
        if verbose: print("\t{}".format(movie_id))
        id_history = {}
        # find matches
        for frame in json[movie_id]:
            for kind in frame.keys():
                for box in frame[kind]:
                    id = box["id"]
                    id_history[id] = id_history[id] + 1 if id in id_history else 1
        # delete low count matches
        ids_to_remove = [key for key in id_history.keys() if id_history[key] < min_required_occurence_count]
        tmp[movie_id] = ids_to_remove

    filter_out_ids(json, tmp, verbose)


def filter_out_ids(json, movie_ids, verbose: False):
    """
    Removes objects with IDs in each video

    Params:
      json
        represents json data on which the operation will be applied
      movie_ids : dict<movie_id, list<ids>>
        IDs to be removed. Keys are movie ids, values are lists of indices to be removed.
      verbose
        print progress comments, optional (False)

    Returns:
        Nothing.

    The changes are done directly over the passed JSON object. There is no local copy created.
    """
    removed_counter = 0
    for movie_id in movie_ids.keys():
        if verbose: print("For movie", movie_id, "removing ids:", movie_ids[movie_id])
        if movie_id not in json: raise KeyError("JSON does not contain data for movie {}".format(movie_id))
        for frame in json[movie_id]:
            for kind in frame.keys():
                # to_rem_boxes = filter(lambda q: q["id"] in movie_ids[movie_id], frame[kind])
                to_rem_boxes = [box for box in frame[kind] if box["id"] in movie_ids[movie_id]]
                for to_rem_box in to_rem_boxes:
                    frame[kind].remove(to_rem_box)
                    removed_counter += 1
    print('removed:', removed_counter)





MAX_FRAME_DISTANCE_TO_INTERPOLATE = 3  # ENG: best value for car is 3-4, for pedestrian is 3
MAX_REMEMBERED_AGE = 10  # change this only if you know what are you doing


class BoxToInterpolate:
    def __init__(self, id, type, frame_index_before, frame_index_after):
        self.id = id
        self.type = type
        self.frame_index_before = frame_index_before
        self.frame_index_after = frame_index_after

    def get_missing_frames_length(self):
        return self.frame_index_after - self.frame_index_before


def filter_by_not_long_gap(interpolations, maximal_gap):
    ret = []
    long = []
    for interpolation in interpolations:
        if interpolation.get_missing_frames_length() <= maximal_gap:
            ret.append(interpolation)
        else:
            long.append(interpolation)

    #if len(long) > 0:
        #print("\t\tafter interpolation filtering, {} records has gap greater than {}".format(
        #    len(long), maximal_gap))
    return ret


def list_first(lst, lambda_selector):
    for item in lst:
        if lambda_selector(item):
            return item
    return None


def interpolate_boxes(box_before, box_after, gap):
    box_before = np.array(box_before)
    box_after = np.array(box_after)
    diff = box_after - box_before
    diff = diff / gap
    curr = box_before.copy()
    ret = []
    for i in range(gap - 1):
        curr = curr + diff
        tmp = curr
        tmp = list(map(int, tmp))
        ret.append(tmp)
    return ret


def generate_interpolations(frames):
    ret = []
    id_age = {}
    id_type = {}
    for frame_index, frame in enumerate(frames):
        for type in frame.keys():
            # if type == 'Car': continue
            for box in frame[type]:
                id = box["id"]
                if id not in id_age:
                    # new box id
                    id_age[id] = 0
                    id_type[id] = type
                elif id_type[id] == type:
                    if id_age[id] > 1:
                        # box was missing in previous frame(s)
                        bti = BoxToInterpolate(id, type, frame_index - id_age[id], frame_index)
                        # print(id, type, frame_index - id_age[id], frame_index)
                        ret.append(bti)
                    id_age[id] = 0
        # age all boxes:
        for id in id_age.keys():
            id_age[id] = id_age[id] + 1
        ids_to_delete = [id for id in id_age.keys() if id_age[id] > MAX_REMEMBERED_AGE]
        for id in ids_to_delete:
            del id_age[id]

    return ret


def apply_interpolations(interpolations, json_data):
    for interpolation in interpolations:
        id = interpolation.id
        gap = interpolation.get_missing_frames_length()
        tmp = json_data[interpolation.frame_index_before][interpolation.type]
        box_before = list_first(tmp, lambda q: q["id"] == id)["box2d"]
        tmp = json_data[interpolation.frame_index_after][interpolation.type]
        box_after = list_first(tmp, lambda q: q["id"] == id)["box2d"]

        boxes_res = interpolate_boxes(box_before, box_after, gap)

        # !!!!!!!!!!!!!!!PETRUV BORDEL!!!!!!!!!!!!!!!!!!
        if interpolation.type == 'Car':
            thresh = 130  # ENG best value, anything above 100 is fine
        else:
            thresh = 30  # pedestrian ENG: 40 is best value, anything between 30 and 50 is fine
        if math.sqrt(pow(box_before[0] - box_after[0], 2) + pow(box_before[1] - box_after[1], 2)) > thresh:
            continue

        for xx in range(0, len(boxes_res)):
            act_box = boxes_res[xx]


            obj = {
                "id": id,
                "box2d": list(map(int, act_box))
            }
            frame_index = interpolation.frame_index_before + xx + 1
            if interpolation.type not in json_data[frame_index]:
                json_data[frame_index][interpolation.type] = []
            json_data[frame_index][interpolation.type].append(obj)

def solve_missing(ss):
    interpolations = generate_interpolations(ss[next(iter(ss))])
    interpolations = filter_by_not_long_gap(interpolations, MAX_FRAME_DISTANCE_TO_INTERPOLATE)
    apply_interpolations(interpolations, ss[next(iter(ss))])
    return ss




