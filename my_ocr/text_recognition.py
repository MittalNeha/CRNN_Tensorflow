import argparse
import os.path as ops
import subprocess

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

import sys, os
sys.path.insert(0, os.getcwd())

from tools.test_shadownet import *

min_conf = 0.5
FFMPEG_CMD = "ffmpeg -f rawvideo -pixel_format yuyv422 -video_size 848x477 -i"



def init_args_recog():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,
                        help='image to be read')
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?', const=True,
                        help='Whether to display images')
    parser.add_argument("-p", "--padding", type=float, default=0.0,
                                                help="amount of padding to add to each border of ROI")

    return parser.parse_args()

def decode_predictions(scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
                # extract the scores (probabilities), followed by the
                # geometrical data used to derive potential bounding box
                # coordinates that surround text
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                # loop over the number of columns
                for x in range(0, numCols):
                        # if our score does not have sufficient probability,
                        # ignore it
                        if scoresData[x] < min_conf:
                                continue

                        # compute the offset factor as our resulting feature
                        # maps will be 4x smaller than the input image
                        (offsetX, offsetY) = (x * 4.0, y * 4.0)

                        # extract the rotation angle for the prediction and
                        # then compute the sin and cosine
                        angle = anglesData[x]
                        cos = np.cos(angle)
                        sin = np.sin(angle)

                        # use the geometry volume to derive the width and height
                        # of the bounding box
                        h = xData0[x] + xData2[x]
                        w = xData1[x] + xData3[x]

                        # compute both the starting and ending (x, y)-coordinates
                        # for the text prediction bounding box
                        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                        startX = int(endX - w)
                        startY = int(endY - h)

                        # add the bounding box coordinates and probability score
                        # to our respective lists
                        rects.append((startX, startY, endX, endY))
                        confidences.append(scoresData[x])

        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

def get_text_boxes(image, W, H):
        # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    return boxes

if __name__ == '__main__':
    """
    
    """
    # init images
    args = init_args_recog()

    # detect text
    image_name = args.image
    print("Find the text in the image " + args.image)

    file_ext = image_name[-4:]
    if(file_ext=='yuyv'):
        new_name = image_name[:-4]+'png'
        print('converting image to PNG format')
        cmd = FFMPEG_CMD +" "+ image_name + " "+ new_name
        #p = subprocess.Popen(['ls',  '-l', '../'])
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        print(p.communicate())
        image_name = new_name

    image = cv2.imread(image_name)
    orig = image.copy()

    #Crop the image for Netflix use case 480, 160
    W=480
    H=224
    image = image[0:224, 0:480]

    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (W*2, H*2)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH), cv2.INTER_AREA)
    (H, W) = image.shape[:2]

    boxes = get_text_boxes(image, W,H)

    #save the image
    #Draw rectangles
    text_boxes = image.copy()
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(text_boxes, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imwrite(image_name[:-4]+'_out.jpg', text_boxes)

    #Order the boundboxes
    xSorted = boxes[np.argsort(boxes[:, 1]), :]
    print("BEFORE:::")
    print(xSorted)

    n_row = 0
    boxes_order = []
    for i,box in enumerate(xSorted):
        if i == 0:
            box = np.concatenate((box,np.array([n_row])))
            print(box)
            boxes_order.append(box)
            continue
        if((box[1] - xSorted[i-1][1] < 4) and (box[1] - xSorted[i-1][1] > -4)):
            box = np.concatenate((box,np.array([n_row])))
            boxes_order.append(box)
        else:
            n_row = n_row+1
            box = np.concatenate((box,np.array([n_row])))
            boxes_order.append(box)
    boxes = sorted(boxes_order , key=lambda k: [k[4], k[0]])
    print("AFTER::")
    print(boxes)

    #Read text for each box
    nRow=0
    out = ""
    for (startX, startY, endX, endY, row) in boxes:
        #cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        print("Reading text {}, {} ".format(startX, startY))

        #add margin to boundbox
        dX = int((endX - startX) * args.padding)
        dY = int((endY - startY) * args.padding)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(newW, endX + (dX * 2))
        endY = min(newH, endY + (dY * 2)) 

        text_read = recognize_image(
          image[startY:endY, startX:endX], 
          weights_path=args.weights_path,
          char_dict_path=args.char_dict_path,
          ord_map_dict_path=args.ord_map_dict_path,
          is_vis=args.visualize)

        if row == nRow:
            out = out + text_read + " "
        else:
            nRow = row
            out = out + '\n' + text_read + " "

    print("OUTPUT")
    print(out)
    #Draw rectangles
    #for (startX, startY, endX, endY) in boxes:
    #    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)


    # show the output image
    #cv2.imshow("Text Detection", image)
    #cv2.waitKey(0)

