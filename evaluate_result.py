import os
import csv
from pycm import *
all_lbl  = []
all_pred = []
with open('faster_r_cnn_pred.csv',"r") as csv_file:
    csvReader = csv.reader(csv_file)
    header = csvReader.next()
    imgIdx = header.index("img_id")
    resnetIdx = header.index("resnet_pred")
    resnet_scoreIdx = header.index("resnet_score")
    frcnnIdx = header.index("frcnn_pred")
    frcnn_scoreIdx = header.index("frcnn_score")
    labelIdx = header.index('label')
    # with open('final_result.csv',"w") as fo:
    for line in csvReader:
        all_lbl.append(line[labelIdx])
        if line[resnetIdx] in ['dyed-lifted-polyps', 'dyed-resection-margins'] and line[frcnnIdx] not in ['polyps','null']:
            all_pred.append(line[frcnnIdx])
            continue
        # if line[frcnnIdx] in ['normal-z-line']:
        #     all_pred.append(line[frcnnIdx])
        #     continue
        all_pred.append(line[resnetIdx])
        
    # print(count)
    cm = ConfusionMatrix(actual_vector=all_lbl, predict_vector=all_pred)
    with open('final_result_eval.txt', "w") as fo:
        fo.write(str(cm))
        fo.close()
    with open('final_result.csv', "w") as fo:
        with open('faster_r_cnn_pred.csv') as fi:
            lines = fi.readlines()
            fo.write(lines[0].strip()+',final_pred\n')
            for i, line in enumerate(lines[1:]):
                fo.write(line.strip() + ',' + all_pred[i] + '\n')
            fi.close()
        fo.close()
