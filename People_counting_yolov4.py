from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import dlib
import time 
import cv2

import os
import sys
import socket
import json

import logging as log
import paho.mqtt.client as mqtt

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
TOPIC = "people_counter_python"
MQTT_HOST = IPADDRESS
MQTT_PORT = 1884
MQTT_KEEPALIVE_INTERVAL = 60

CONFIG_FILE = '../resources/config.json'

use_cuda = False #True
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_cv2_camera(cfgfile, weightfile):
    #connecting to MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    client.subscribe(TOPIC)

    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("TownCentreXVID.avi")#"Pedestrian_Detect_2_1_1.mp4")
    #cap.set(3, 1280)
    #cap.set(4, 720)

    num_classes = m.num_classes
    print(num_classes)
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    count = 0 #total number of frames 
    total_count = 0 #total number of people counted
    last_count = 0


    while True:
        #getting FPS of the video - to be used for time calculation
        #print(cap.get(cv2.CAP_PROP_FPS))
        count += 1
        new_boxes = []

        ret, img = cap.read()
        if count//60:
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            start = time.time()
            boxes = do_detect(m, sized, 0.6, 0.6, use_cuda)
            finish = time.time()
            print('Predicted in %f seconds.' % (finish - start))

            for x in range(len(boxes[0])):
                if boxes[0][x][-1] == 0:
                    new_boxes.append(boxes[0][x])
            print("FPS : {}".format(1/(finish -  start)))

            if len(new_boxes) > last_count:
                start_time = time.time()
                total_count = total_count + abs(len(new_boxes) - last_count)

            # Person duration in the video is calculated
            if len(new_boxes) < last_count:
                duration = int(time.time() - start_time)

            last_count = len(new_boxes)

            print("Current number of people: {}".format(len(new_boxes)))
            print("Total number of people: {}".format(total_count))
            #print("Duration : {}".format(duration)) if duration else print("")

            client.publish("person", json.dumps({"total": total_count}))
            client.publish("person/duration",json.dumps({"duration": duration}))
       
        result_img = plot_boxes_cv2(img, new_boxes, savename=None, class_names=class_names)
        client.publish("person", json.dumps({"count": len(new_boxes)}))

        #total_count = total_count + (len(new_boxes) - total_count) if len(new_boxes)> total_count else total_count

        cv2.imshow('People Counter', result_img)
        cv2.waitKey(27)
        client.disconnect()

    cap.release()


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2_camera(args.cfgfile, args.weightfile)