# USAGE
'''
python deep_learning_with_opencv.py --image images/jemma.png \
	--prototxt bvlc_googlenet.prototxt \
	--model bvlc_googlenet.caffemodel --labels synset_words.txt
'''

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import sys

def classify(args):
    # load the input image from disk
    image = cv2.imread(args.image)
    # load the class labels from disk
    rows = open(args.labels).read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

    # our CNN requires fixed spatial dimensions for our input image(s)
    # so we need to ensure it is resized to 224x224 pixels while
    # performing mean subtraction (104, 117, 123) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 224, 224)
    print(f"[INFO] for model {args.name}: scale factor is {args.scale_factor} and mean is {args.mean}")
    blob = cv2.dnn.blobFromImage(image, args.scale_factor, (224, 224), args.mean)

    # load our serialized model from disk
    print("[INFO] loading model...", file=sys.stderr)
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    total_time = end-start
    print("[INFO] classification took {:.5} seconds".format(total_time), file=sys.stderr)
    flops = net.getFLOPS((1,3,224,224))

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    preds = preds.reshape(1, 1000)
    idxs = np.argsort(preds[0])[::-1]

    result_labels = []
    probabilities = []
    # loop over the top-5 predictions and display them
    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
            cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if i < 5:
            # display the predicted label + associated probability to the
            # console
            print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]), file=sys.stderr)
        result_labels.append(classes[idx])
        probabilities.append(preds[0][idx])

    return image, total_time, result_labels, probabilities, flops


def parse_args():
    import os
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-l", "--labels", required=True,
                    help="path to ImageNet labels (i.e., synsets)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print additional information to stderr")
    args = ap.parse_args()
    args.name,ext = os.path.splitext(os.path.basename(args.model))
    args.scale_factor = .017 if "mobilenet" in args.model or "shufflenet" in args.model else 1
    args.mean = (103.94, 116.78, 123.68) if "mobilenet" in args.model or "shufflenet" in args.model else (104, 117, 123)
    return args

if __name__ == "__main__":
    args = parse_args()
    image, total_time, labels, probabilities, flops = classify(args)

    # display the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
