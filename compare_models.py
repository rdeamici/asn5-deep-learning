from p5_classification import classify as p5_classify
import os, sys

class Classifier:
    def __init__(self, name):
        self.path_to_classifiers = "caffe_models/image_classification/"
        self.path_to_images = "caffe_models/images/"
        self.labels = "synset_words.txt"
        self.name = name
        self.path_to_classifier = os.path.join(self.path_to_classifiers, name)
        self.prototxt = None
        self.model = None
        self.image = None
        self.result_image = None
        self.total_time = None
        self.cl_labels = None
        self.probabilities = None

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
        self.result_image, self.total_time, self.cl_labels, self.probabilities = p5_classify(self)

    def invalid(self):
        return None in (self.prototxt, self.model, self.image, self.labels)


def compare():
    classifiers = ["alexnet","googlenet","mobilenet","resnet_101","shufflenet"]
    path_to_images = "caffe_models/images/"
    results = []

    for image in os.listdir(path_to_images):
        for classifier in classifiers:
            classifier = Classifier(classifier)
            classifier.input_image = image

            for file in os.listdir(classifier.path_to_classifier):
                if "prototxt" in file:
                    classifier.prototext = file
                elif "caffemodel" in file:
                    classifier.input_model = file

            if classifier.invalid():
                print("*********FATAl ERROR*********")
                issue_in_classifier_dir = False
                if classifier.prototxt is None:
                    print("prototxt is None")
                    issue_in_classifier_dir = True
                if classifier.model is None:
                    print("model is None")
                    issue_in_classifier_dir = True
                if issue_in_classifier_dir:
                    print(f"available files in '{classifier.path_to_classifier}'")
                    for f in os.listdir(classifier.path_to_classifier):
                        print(f)
                if classifier.labels is None:
                    print("labels is None")
                if classifier.image is None:
                    print("classifier.image is None. This shouldn't be possible")
                    print(f"available images found in '{path_to_images}'")
                    for i in os.listdir(path_to_images):
                        print(i)
                sys.exit(1)
            
            classifier.classify()
            results.append(classifier)

    return results

def write_results_to_file(results):
    header = ["image", "classifier","best label", "probability","total_time"]
    with open("model_results.csv", "w") as f:
        print(",".join(header), file=f)
    
    delimiter = ", "
    with open("model_results.csv", "a") as f:
        for classifier in results:
            f.write(os.path.basename(classifier.image))
            f.write(delimiter+classifier.name)
            f.write(delimiter+classifier.cl_labels[0])
            f.write(delimiter+"{:.5}".format(classifier.probabilities[0]))
            f.write(delimiter+"{:.5}".format(classifier.total_time))
            f.write("\n")


def main():
    results = compare()
    write_results_to_file(results)

if __name__ == "__main__":
    main()