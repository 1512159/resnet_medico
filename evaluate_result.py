import os
import csv
from pycm import *
from my_reporter_final import EvalReporter
all_lbl  = []
all_pred = []
all_img_path = []
all_prob = []
with open('faster_r_cnn_pred.csv',"r") as csv_file:
    csvReader = csv.reader(csv_file)
    header = csvReader.next()
    imgIdx = header.index("img_id")
    # resnetIdx = header.index("resnet_pred")
    # resnet_scoreIdx = header.index("resnet_score")
    frcnnIdx = header.index("frcnn_pred")
    labelIdx = header.index('label')
    # with open('final_result.csv',"w") as fo:
    for line in csvReader:
        all_lbl.append(line[labelIdx])
        all_pred.append(line[frcnnIdx])
        all_img_path.append(line[imgIdx])
        all_prob.append('')
        
    cm = ConfusionMatrix(actual_vector=all_lbl, predict_vector=all_pred)
    print(str(cm).split('\n')[43])
    
    #write eval report
    with open('final_result_eval.txt', "w") as fo:
        fo.write(str(cm))
        fo.close()
    
    #write html file vor visualization
    reporter = EvalReporter(all_img_path, all_pred, all_lbl, all_prob)
    reporter.write_html_file('final_visualization.html')
    
    # with open('final_ev.csv', "w") as fo:
    #     with open('faster_r_cnn_pred.csv') as fi:
    #         lines = fi.readlines()
    #         fo.write(lines[0].strip()+',final_pred\n')
    #         for i, line in enumerate(lines[1:]):
    #             fo.write(line.strip() + ',' + all_pred[i] + '\n')
    #         fi.close()
    #     fo.close()

    