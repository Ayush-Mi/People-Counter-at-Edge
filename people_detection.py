from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import dlib
use_cuda = False #True

def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("Pedestrian_Detect_2_1_1.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    print(num_classes)
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    count = 0
    track_obj = False
    trackers = []
    rect = []

    while True:
        #getting FPS of the video - to be used for time calculation
        #print(cap.get(cv2.CAP_PROP_FPS))
        count += 1
        new_boxes = []

        ret, img = cap.read()
        if track_obj == False: #count//5 and (track_obj == False):
            print("not tracking")

            track_obj = True

            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            start = time.time()
            boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
            print(boxes)
            finish = time.time()
            print('Predicted in %f seconds.' % (finish - start))

            for x in range(len(boxes[0])):
                if boxes[0][x][-1] == 0:

                    new_boxes.append(boxes[0][x])
                    print('human detected')

                    tracker = dlib.correlation_tracker()
                    tracker.update(img)
                    rect = dlib.rectangle(new_boxes[0])
                    tracker.start_track(img,rect)
                    trackers.append(tracker)
            
        else:
            print("tracking")
            for tracker in trackers:
                #tracker.update(img)
                position = tracker.get_position()
                startX = int(position.left())
                startY = int(position.top())
                endX = int(position.right())
                endY = int(position.bottom())
                # draw the bounding box from the correlation object tracker
                cv2.rectangle(img, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)

            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            start = time.time()
            boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
            print(boxes)
            finish = time.time()
            print('Predicted in %f seconds.' % (finish - start))

            for x in range(len(boxes[0])):
                if boxes[0][x][-1] == 0:
                    new_boxes.append(boxes[0][x])
            
            if len(trackers) != len(new_boxes):
                track_obj = False

    
        result_img = plot_boxes_cv2(img, new_boxes, savename=None, class_names=class_names)

        print("Current number of people: {}".format(len(new_boxes)))

        cv2.imshow('People Counter', result_img)
        cv2.waitKey(27)

    cap.release()


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2_camera(args.cfgfile, args.weightfile)