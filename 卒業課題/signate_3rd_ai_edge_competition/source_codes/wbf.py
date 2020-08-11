def wbf(bboxes: list, scores: list, iou_threshold: float, n: int) -> (list, list):
    lists, fusions, confidences = [], [], []
    
    indexes = sorted(range(len(bboxes)), key=scores.__getitem__)[::-1]
    for i in indexes:
        new_fusion = True
        
        for j in range(len(fusions)):
            if iou(bboxes[i], fusions[j]) > iou_threshold:
                lists[j].append(bboxes[i])
                confidences[j].append(scores[i])
                fusions[j] = (
                    sum([l[0] * c for l, c in zip(lists[j], confidences[j])]) / sum(confidences[j]),
                    sum([l[1] * c for l, c in zip(lists[j], confidences[j])]) / sum(confidences[j]),
                    sum([l[2] * c for l, c in zip(lists[j], confidences[j])]) / sum(confidences[j]),
                    sum([l[3] * c for l, c in zip(lists[j], confidences[j])]) / sum(confidences[j]),
                )
                
                new_fusion = False
                break

        if new_fusion:
            lists.append([bboxes[i]])
            confidences.append([scores[i]])
            fusions.append(bboxes[i])
            
        print(lists, fusions, confidences)
            
    confidences = [(sum(c) / len(c)) * (min(n, len(c)) / n) for c in confidences]
    
    return fusions, confidences
    
print(wbf([(10, 10, 20, 20), (15, 10, 25, 20), (15, 15, 25, 25), (30, 30, 40, 40)], [0.8, 0.7, 0.9, 0.5], 0.1, 3))
# ([(13.333333333333332, 11.874999999999998, 23.33333333333333, 21.874999999999996), (30, 30, 40, 40)], [0.8000000000000002, 0.16666666666666666])