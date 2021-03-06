from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2
import os



filename = 'results.txt'

if os.path.exists(filename):
    append_write = 'a'
else:
    append_write = 'w'

# argument parser

# in console:
# python realtime_maskdetection.py -c 0.5 -d 1 -i VIDEO FILE

ap = argparse.ArgumentParser()

# change the confidence threshold of face detection. default is 50%
ap.add_argument("-c", "--confidence", default=0.5,
                help="confidence threshold")

# set the output you want to display the real time inference happening. set to 1 for primary monitor
ap.add_argument("-d", "--display", type=int, default=0,
                help="switch to display image on screen")

# select input video file, if left blank it will just default to RPi camera
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
args = vars(ap.parse_args())

# Load the modelS
# FACE DETECTION MODEL
net = cv2.dnn.readNet('models/test/face-detection-adas-0001.xml', 'models/test/face-detection-adas-0001.bin')
# MASK DETECTION MODEL
mask_net = cv2.dnn.readNet('models/test/mask_resnet50.xml', 'models/test/mask_resnet50.bin')

# SET TARGET DEVICE TO NCS2
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
mask_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# If no video file, use webcam
if not args.get("input", False):
    print("STARTING WEBCAM STREAM...")
    # cap = cv2.VideoCapture(0)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# Else use video file gioven in arg
else:
    print("OPENING VIDEO FILE...")
    vs = cv2.VideoCapture(args["input"])

# errors happened without this timeout.
time.sleep(2)



def predict(frame, net):
    # Prepare input blob and perform an inference
    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()

    predictions = []

    # Draw detected faces on the frame
    for detection in out.reshape(-1, 7):
        conf = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        if conf > args["confidence"]:
            pred_boxpts = ((xmin, ymin), (xmax, ymax))

            # create prediciton tuple and append the prediction to the
            # predictions list
            prediction = (conf, pred_boxpts)
            predictions.append(prediction)

    # return the list of predictions to the calling function
    return predictions


def mask_predict(frame, net):
    # Prepare input blob and perform an inference
    try:
        blob = cv2.dnn.blobFromImage(frame, size=(224, 224))  # change size depending on model
    except Exception as e:
        print(str(e))
        return [-9999]
    net.setInput(blob)
    out = net.forward()

    predictions = out

    # return mask predictions
    print("MASK PREDICTIONS ----> ", predictions)
    return predictions


# loop over frames
frame_count = -1
# start fps counter and timer
fps = FPS().start()
start_time = time.time()

while True:
    faceDetected = "no classifcation"
    
    try:
        frame_count += 1
        #grab the frame from the threaded video stream
        # make a copy of the frame for displaying the result
        frame = vs.read()  # current frame
        frame = frame[1] if args.get("input", False) else frame
        image_for_result = frame.copy()  # copy of current frame for output

        # using NCS2 get the face predictions
        face_predictions = predict(frame, net)

        # loop over face_predictions
        for (i, pred) in enumerate(face_predictions):
            # extract prediction data for readability
            (pred_conf, pred_boxpts) = pred

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if pred_conf > args["confidence"]:
                faceDetected = "classifcation"
                # print prediction to terminal
                print("FACE-DETECTION = Prediction #{}: confidence={}, "
                      "boxpoints={}".format(i, pred_conf,
                                            pred_boxpts))

                # store the predicted face coordinates so that:
                # x1,y1 represents top left point
                # x2,y2 represents bottom right point
                x1, y1, x2, y2 = pred_boxpts[0][0], pred_boxpts[0][1], pred_boxpts[1][0], pred_boxpts[1][1]

                # using stored coordinates, crop the frame using slicing to get ROI
                # using frame[y1:y2, x1:x2].
                # expand ROI frame slightly by lowering the values of y1 and x1 and increasing y2,x2.
                # this insures the face is not too cropped in
                Frame_ROI = frame[y1 - 50:y2 + 50, x1 - 25:x2 + 25]

                # get mask predictions with new cropped fram from NCS
                mask_predictions = mask_predict(Frame_ROI, mask_net)

                # check if we should show the prediction data
                # on the frame
                if args["display"] > 0:
                    # build a label
                    if mask_predictions[0][0] < 0.0:
                        label = "No_Mask: {:.2f}%".format(pred_conf * 100)
                        color = (0, 0, 255)
                    else:
                        label = "MASK: {:.2f}%".format(pred_conf * 100)
                        color = (255,0,0)

                    # extract information from the prediction boxpoints
                    (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
                    (startX, startY) = (ptA[0], ptA[1])
                    y = startY - 15 if startY - 15 > 15 else startY + 15

                    # display the rectangle and label text
                    cv2.rectangle(image_for_result, ptA, ptB,
                                    (255, 0, 0), 2)
                    cv2.putText(image_for_result, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                    

        # check if we should display the frame on the screen
        # with prediction data (you can achieve faster FPS if you
        # do not output to the screen)
        if args["display"] > 0:
            # display the frame to the screen
            cv2.imshow("Output", image_for_result)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # update the FPS counters
        fps.update()
        
        if  frame_count% 5 == 0:
            
            
            timer = (time.time() - start_time)
            start_time = time.time() # reset timer
          
            fps_5frames = 5/timer # fps = 5 / amount of time taken for 5 frames
            print(timer, frame_count, fps_5frames)
            resultOut = open(filename,append_write)
            resultOut.write(str(timer) +" "+  str(frame_count) +" "+ str(fps_5frames) + " " + faceDetected + "\n")
            resultOut.close()
       

    # if "ctrl+c" is pressed in the terminal, break from the loop
    except KeyboardInterrupt:
        break

    # if there's a problem reading a frame, break
    except AttributeError:
        break

# stop the FPS counter timer
fps.stop()

# destroy all windows if we are displaying them
if args["display"] > 0:
    cv2.destroyAllWindows()

# if not using a video file, stop the camera stream
if not args.get("input", False):
    vs.stop()

# else, stop the video file
else:
    vs.release()

# display FPS information
print("ELAPSED TIME: {:.2f}".format(fps.elapsed()))
print("AVARAGE FPS: {:.2f}".format(fps.fps()))