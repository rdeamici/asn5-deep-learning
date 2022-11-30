# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages

from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video_in", default=None,
  help="path to input video. Use the camera, /dev/video0, if not specified.")
ap.add_argument("-o", "--video_out", default=None,
  help="path to output video")
ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
  help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
  help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
  help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

if args["video_out"] is not None:
  writer = None
  if args["video_in"] is None:
    print("Error: video_in is required.")
    raise SystemExit

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
# PASCAL VOC dataset: 20 classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
  "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
if args["video_in"] is None:
  print("[INFO] starting video stream: /dev/video0 ...")
  vs = cv2.VideoCapture(0)
else:
  print("[INFO] starting video stream:", args["video_in"], "...")
  vs = cv2.VideoCapture(args["video_in"])

if vs is None or not vs.isOpened():
  print("Error: unable to open video source")
  raise SystemExit

time.sleep(2.0)

ret, frame = vs.read()
print("[INFO] the input frame size (h, w, d) = "+str(frame.shape))

fps = FPS().start()

# loop over the frames from the video stream
while True:
  # grab the frame from the threaded video stream and resize it
  # to have a maximum width of 400 pixels
  ret, frame = vs.read()
  if ret != True:
    break

  # grab the frame dimensions and convert it to a blob
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    0.007843, (300, 300), 127.5)

  # pass the blob through the network and obtain the detections and
  # predictions
  net.setInput(blob)
  detections = net.forward()

  # loop over the detections
  for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
      # extract the index of the class label from the
      # `detections`, then compute the (x, y)-coordinates of
      # the bounding box for the object
      idx = int(detections[0, 0, i, 1])
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")

      # draw the prediction on the frame
      label = "{}: {:.2f}%".format(CLASSES[idx],
        confidence * 100)
      cv2.rectangle(frame, (startX, startY), (endX, endY),
        COLORS[idx], 2)
      y = startY - 15 if startY - 15 > 15 else startY + 15
      cv2.putText(frame, label, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

  if args["video_out"] is None:
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
      break
  else:
    # check if the video writer is None
    if writer is None:
      print("[INFO] Path to Output Video:", args["video_out"])
      # initialize our video writer
      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
      writer = cv2.VideoWriter(args["video_out"], fourcc, 30,
        (frame.shape[1], frame.shape[0]), True)
    # write the output frame to disk
    writer.write(frame)

  # update the FPS counter
  fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
if args["video_out"] is not None:
  writer.release()
vs.release()