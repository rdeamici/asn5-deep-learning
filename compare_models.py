from p5_classification import classify as p5_classify
import os, sys

path_to_classifiers = "caffe_models/image_classification/"
path_to_images = "caffe_models/images/"

class Classifier:
    def __init__(self, name, scale_factor, mean):
        self.path_to_images = path_to_images
        self.labels = "synset_words.txt"
        self.name = name
        self.scale_factor = scale_factor
        self.mean = mean

        self.path_to_classifier = os.path.join(path_to_classifiers, name)
        self.prototxt = None
        self.model = None
        self.image = None
        self.result_image = None
        self.total_time = None
        self.cl_labels = None
        self.probabilities = None
        self.flops = None

    @property
    def prototext(self):
        return self.prototxt
    @prototext.setter
    def prototext(self, filename):
        self.prototxt = os.path.join(self.path_to_classifier, filename)


    @property
    def input_model(self):
        return self.model
    @input_model.setter
    def input_model(self, filename):
        self.model = os.path.join(self.path_to_classifier,filename)

    @property
    def input_image(self):
        return self.image
    @input_image.setter
    def input_image(self, filename):
        self.image = os.path.join(self.path_to_images, filename)

    def classify(self):
        self.result_image, self.total_time, self.cl_labels, self.probabilities, self.flops  = p5_classify(self)
        print("[INFO] flops",self.flops, file=sys.stderr)
        print(str(self.flops), file=sys.stderr)

    def invalid(self):
        return None in (self.prototxt, self.model, self.image, self.labels)


def compare():
    results = []

    for image in os.listdir(path_to_images):
        for classifiername in os.listdir(path_to_classifiers):
            if os.path.isfile(os.path.join(path_to_classifiers,classifiername)):
                continue
            
            if "shufflenet" in classifiername or "mobilenet" in classifiername:
                scale_factor = 0.017
                mean = (103.94, 116.78, 123.68)
            else:
                scale_factor = 1
                mean = (104, 117, 123)
            
            classifier = Classifier(classifiername, scale_factor, mean)
            classifier.input_image = image

            for file in os.listdir(classifier.path_to_classifier):
                if "152" in file:
                    continue
                if "prototxt" in file:
                    classifier.prototext = file
                elif "caffemodel" in file:
                    classifier.input_model = file

            if classifier.invalid():
                print("[ERROR] *********FATAl ERROR*********", file = sys.stderr)
                issue_in_classifier_dir = False
                if classifier.prototxt is None:
                    print("[ERROR] prototxt is None", file = sys.stderr)
                    issue_in_classifier_dir = True
                if classifier.model is None:
                    print("[ERROR] model is None", file = sys.stderr)
                    issue_in_classifier_dir = True
                if issue_in_classifier_dir:
                    print(f"[ERROR] available files in '{classifier.path_to_classifier}'", file = sys.stderr)
                    for f in os.listdir(classifier.path_to_classifier):
                        print(f, file = sys.stderr)
                if classifier.labels is None:
                    print("[ERROR] labels is None", file = sys.stderr)
                if classifier.image is None:
                    print("[ERROR] classifier.image is None. This shouldn't be possible", file = sys.stderr)
                    print(f"[ERROR] available images found in '{path_to_images}'", file = sys.stderr)
                    for i in os.listdir(path_to_images):
                        print(i, file = sys.stderr)
                print("[ERROR] *****SKIPPING CLASSIFIER*****", file = sys.stderr)
            else:
                print(f"[INFO] CLASSIFIER NUMBER {len(results)+1}", file=sys.stderr)
                classifier.classify()
                results.append(classifier)

    return results

def write_results_to_file(results):
    delimiter = ","
    header = ["image", "classifier","best guess", "probability","total_time", "flops"]
    with open("model_results.csv", "w") as f:
        f.write(delimiter.join(header))
        f.write("\n")
        for classifier in results:
            f.write(os.path.basename(classifier.image))
            f.write(delimiter+classifier.name)
            f.write(delimiter+classifier.cl_labels[0])
            f.write(delimiter+"{:.5}".format(classifier.probabilities[0]))
            f.write(delimiter+"{:.5}".format(classifier.total_time))
            f.write(delimiter+str(classifier.flops))
            f.write("\n")

    image_header = ["classifier", "best guess", "probability", "correct label", "probability", "total_time", "flops"]
    for image in os.listdir(path_to_images):
        filename = os.path.basename(image)
        image_name, ext = os.path.splitext(filename)
        filename = os.path.join(image_name+"_results.csv")
        with open(filename,"w") as f:
            if "cat" in image:
                h = image_header[:3]+image_header[5:]
            else:
                h = image_header
            f.write(delimiter.join(h)+"\n")
            for classifier in results:
                if image_name in classifier.image:
                    synset_label = image_name.replace("_"," ")
                    # jemma is the name of a beagle
                    if "jemma" in synset_label:
                        synset_label = "beagle"
                    elif "eagle" in synset_label:
                        synset_label = "bald eagle"
                    # cat.jpg is the only one without a specific result we are looking for
                    elif "cat" in synset_label:
                        synset_label = None
                    f.write(classifier.name)
                    f.write(delimiter+classifier.cl_labels[0])
                    f.write(delimiter+"{:.5}".format(classifier.probabilities[0]))

                    if synset_label is not None:
                        for idx, label in enumerate(classifier.cl_labels):
                            if label == synset_label:
                                f.write(delimiter+label)
                                f.write(delimiter+"{:.5}".format(classifier.probabilities[idx]))

                    f.write(delimiter+"{:.5}".format(classifier.total_time))
                    f.write(delimiter+str(classifier.flops))
                    f.write("\n")


def main():
    results = compare()
    write_results_to_file(results)

if __name__ == "__main__":
    main()
