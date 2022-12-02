from p5_classification import classify as p5_classify
import os

class Classifier:
    def __init__(self, name):
        self.path_to_classifiers = "caffe_models/image_classification/"
        self.path_to_images = "caffe_models/images/"
        self.labels = "synset_words.txt"
        
        self.path_to_classifier = self.path_to_classifiers+name
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
        self.prototxt = self.path_to_classifier+filename


    @property
    def input_model(self):
        return self.model
    @input_model.setter
    def input_model(self, filename):
        self.model = self.path_to_classifier+filename

    @property
    def input_image(self):
        return self.image
    @input_image.setter
    def input_image(self, filename):
        self.image = self.path_to_images+filename

    def classify(self):
        self.result_image, self.total_time, self.cl_labels, self.probabilities = p5_classify(self)


def compare():
    classifiers = ["alexnet","googlenet","mobilenet","resnet_101","shufflenet"]
    path_to_images = "caffe_models/images/"
    results = []

    for image in os.listdir(path_to_images):
        for classifier in classifiers:
            classifier = Classifier(classifier)
            classifier.input_image = image
            for file in os.listdir(classifier.path_to_classifier):
                if "prototext" in file:
                    classifier.prototext = file
                elif "caffemodel" in file:
                    classifier.input_model = file

                classifier.image = image
                classifier.classify()
                results.append(classifier)
    return results

def write_results_to_file(results):
    header = ["image", "classifier","label", "probability","total_time"]
    with open("model_results.csv", "w") as f:
        print(",".join(header), file=f)
    
    delimiter = ", "
    with open("model_results.csv", "a") as f:
        for classifier in results:
            for cl_label, probability in zip(classifier.cl_labels, classifier.probabilities):
                f.write(classifier.image)
                f.write(delimiter+classifier.name)
                f.write(delimiter+cl_label)
                f.write(delimiter+probability)
                f.write(delimiter+classifier.total_time)
                f.write("\n")


def main():
    results = compare()


if __name__ == "__main__":
    main()