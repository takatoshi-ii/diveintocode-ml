import cv2 as cv
import numpy as np
import os
import time

from keras.models import load_model

import efficientnet.keras as efn

import yolos.yolo_v4_wo_poly_multiscale as yolo
import yolos.yolo_v4_wo_poly_multiscale_v4 as yolo_epic
import yolos.yolo_v4_full_res as yolo_full_res

import support as support

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_resource_variables()


class ScoringService(object):  
    model = []

    @classmethod
    def get_model(cls, model_path='../model'):
        """Get model method
        Args: model_path (str): Path to the trained model directory.
        Returns: bool: The return value. True for success, False otherwise."""

        cls.model.append(
            yolo.YOLO(
                model_path=model_path+'/chleba.h5',
                model_image_size=(448, 864),
                score=0.5,
                iou=0.5,
                anchors_path='yolos/yolo_anchors.txt',
                classes_path='yolos/yolo_classes.txt'
            )
        )
        cls.model.append(
            yolo.YOLO(
                model_path=model_path+'/chleba2.h5',
                model_image_size=(352, 704),
                score=0.4,
                iou=0.5,
                anchors_path='yolos/yolo_anchors.txt',
                classes_path='yolos/yolo_classes.txt'
            )
        )
        cls.model.append(
            yolo_epic.YOLO(
                model_path=model_path+'/chleba3.h5',
                model_image_size=(224, 448),
                score=0.4,
                iou=0.5,
                anchors_path='yolos/yolo_anchors.txt',
                classes_path='yolos/yolo_classes.txt'
            )
        )
        cls.model.append(
            yolo_full_res.YOLO(
                model_path=model_path+'/chleba4.h5',
                model_image_size=(960, 1952),
                score=0.8,
                iou=0.5,
                anchors_path='yolos/yolo_anchors_full_res.txt',
                classes_path='yolos/yolo_classes_pedest_only.txt'
            )
        )
        cls.model.append(
            yolo_epic.YOLO(
                model_path=model_path + '/chleba5.h5',
                model_image_size=(544, 1120),
                score=0.5,
                iou=0.5,
                anchors_path='yolos/yolo_anchors.txt',
                classes_path='yolos/yolo_classes.txt'
            )
        )

        cls.model.append(load_model(model_path+'/classifier.h5'))
        cls.model[-1]._make_predict_function()
        cls.model.append(load_model(model_path + '/refiner.h5'))
        cls.model[-1]._make_predict_function()
        cls.model.append(load_model(model_path + '/matcher.h5'))
        cls.model[-1]._make_predict_function()
        cls.model.append(load_model(model_path + '/refiner_pedest.h5'))
        cls.model[-1]._make_predict_function()

        #TODO: return true/false    
        return True
            

    @classmethod
    def predict(cls, input):
        predictions = []
        cap = cv.VideoCapture(input)
        fname = os.path.basename(input)

        act_ids = []
        last_ids = []
        act_boxes = []
        last_boxes = []
        act_classes = []
        last_classes = []
        track_ids = []
        track_boxes = []
        trackers = []
        last_wh = []
        act_wh = []
        act_id = 0
        all_ids = []
        
              
        
        nr_image = 0
        start = time.time()
        first = True
        while True:
        
            ret, frame = cap.read()
            if not ret: break

            imgcv = frame[100:1050, ...]


            box, score, classs = cls.model[0].detect_image(imgcv)
            box2, score2, classs2 = cls.model[1].detect_image(imgcv)
            box3, score3, classs3 = cls.model[2].detect_image(imgcv)
            box4, score4, classs4 = cls.model[3].detect_image(imgcv)
            classs4[...] = 1
            box5, score5, classs5 = cls.model[4].detect_image(imgcv)
            box5    = box5[classs5 == 1]
            score5  = score5[classs5 == 1]
            classs5 = classs5[classs5 == 1]



            boxes_append = []
            scores_append = []
            classs_append = []
            if len(box):
                boxes_append.append(box)
                scores_append.append(score)
                classs_append.append(classs)
            if len(box2):
                boxes_append.append(box2)
                scores_append.append(score2)
                classs_append.append(classs2)
            if len(box3):
                boxes_append.append(box3)
                scores_append.append(score3)
                classs_append.append(classs3)
            if len(box4):
                boxes_append.append(box4)
                scores_append.append(score4)
                classs_append.append(classs4)
            if len(box5):
                boxes_append.append(box5)
                scores_append.append(score5)
                classs_append.append(classs5)


            predictions.append({})

            if len(boxes_append):
                box = np.concatenate(boxes_append)
                score = np.concatenate(scores_append)
                classs = np.concatenate(classs_append)


            deleted_boxes = []
            box, score, classs = support.order_boxes_and_scores_by_score(box, score, classs)
            box, score, classs = support.solve_intersect_avg(box, score, classs, 0.45)
            box, score, classs = support.solve_intersect_avg(box, score, classs, 0.45)  # yes, it has a reason why we apply it multiple times
            box, score, classs, deleted_boxes = support.filter_boxes(box, score, classs, imgcv.shape[1] - 1, imgcv.shape[0] - 1, 900, deleted_boxes)

            # correcting classes by a classifier
            box, score, classs = support.correct_by_classifier(box, score, classs, imgcv, cls.model[5])

            # refine the detections with precise refiner
            box = support.correct_by_refiner(box, imgcv, cls.model[6])
            box, score, classs, deleted_boxes = support.filter_boxes(box, score, classs, imgcv.shape[1] - 1, imgcv.shape[0] - 1, 1024, deleted_boxes)
            box, score, classs = support.solve_intersect_avg(box, score, classs, 0.45)

            box = support.correct_by_refiner(box, imgcv, cls.model[6])
            box, score, classs, deleted_boxes = support.filter_boxes(box, score, classs, imgcv.shape[1] - 1, imgcv.shape[0] - 1, 1024, deleted_boxes)
            box, score, classs = support.solve_intersect_avg(box, score, classs, 0.45)

            #refiner specialized on pedestrians
            box[classs == 1] = support.correct_by_refiner(box[classs == 1], imgcv, cls.model[8])
            box, score, classs = support.solve_intersect_avg(box, score, classs, 0.45)
            box, score, classs, deleted_boxes = support.filter_boxes(box, score, classs, imgcv.shape[1] - 1, imgcv.shape[0] - 1, 1024, deleted_boxes)

            box, score, classs = support.correct_by_classifier(box, score, classs, imgcv, cls.model[5])

            # the pedestrians with too big ar are mostly false positive
            box, score, classs = support.filter_aspect_ratio(box, score, classs, type=1, thresh=4.2)

            # we can filter the cars too
            box, score, classs = support.filter_aspect_ratio(box, score, classs, type=0, thresh=2.1)



            if len(box)>0:
                tmp_0 = box[:, 0].copy()
                tmp_2 = box[:, 2].copy()
                box[:, 0] = box[:, 1].copy()
                box[:, 2] = box[:, 3].copy()
                box[:, 1] = tmp_0
                box[:, 3] = tmp_2
                box[:, 0] = np.clip(box[:, 0], 0, imgcv.shape[1] - 1)
                box[:, 1] = np.clip(box[:, 1], 0, imgcv.shape[0] - 1)
                box[:, 2] = np.clip(box[:, 2], 0, imgcv.shape[1] - 1)
                box[:, 3] = np.clip(box[:, 3], 0, imgcv.shape[0] - 1)

            if first:
                first = False
                act_id, last_boxes, last_ids, all_ids = support.init_trackers(cv.cvtColor(imgcv, cv.COLOR_BGR2RGB), box, act_id, cls.model[7], all_ids, classs)
            else:
                act_id, last_boxes, last_ids, all_ids = support.update_trackers(cv.cvtColor(imgcv, cv.COLOR_BGR2RGB), box, act_id, last_boxes, last_ids, cls.model[7], all_ids, classs)


            #0: id; 1: is_used; 2: count_death; 3: vector
            for i in range(len(all_ids)-1, -1, -1):
                if all_ids[i][1] == False: 
                    all_ids[i][2] += 1
                    if all_ids[i][2] > 8:
                        del all_ids[i]

        
            for k in range (0, len(box)):
              cls_name = support.translate_name(classs[k])
              if cls_name not in predictions[-1]:
                  predictions[-1][cls_name] = []   
      
              predictions[-1][cls_name].append({})
              predictions[-1][cls_name][-1]['id'] = str(last_ids[k])
              predictions[-1][cls_name][-1]['box2d'] =  [int(box[k][0]), int(box[k][1] + 100), int(box[k][2]), int(box[k][3] + 100)]
              #cv.rectangle(imgcv, (box[k][0], box[k][1]), (box[k][2], box[k][3]), (255, 0, 255), 2, 1)
              #cv.putText(imgcv, str(last_ids[k]), (int(box[k][0]), int(box[k][1] - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            #cv.imwrite(str(nr_image) + '.jpg', imgcv)

            nr_image += 1
            #print(nr_image)
            
            
        cap.release()
        #print('done in: ', time.time()-start)
        #print(predictions)

        ret = {fname: predictions}
        support.filter_out_ids_with_low_occurrence(ret, 3, False)
        ret = support.solve_missing(ret)
        
        return ret
