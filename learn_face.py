import cv2
import os
import zipfile
import shutil
from detection import Detection
from extract_embeddings import ExtractEmbeddings
from train_model import TrainModel


class LearnFace:

    def __init__(self, dataset, proto, model, embeddings, recognizer, le, embedding_model):
        self.dataset = dataset
        self.proto = proto
        self.model = model
        self.embeddings = embeddings
        self.recognizer = recognizer
        self.le = le
        self.embedding_model = embedding_model

    def learn_face(self, nb_image, user):
        d = Detection(self.proto, self.model)
        face_images = d.get_face(nb_image, 0.8)
        # place where images to train the recognizer on are stored
        path_to_images = "temp/dataset/%s" % user
        if not os.path.exists(path_to_images):
            os.makedirs(path_to_images)
        i = 0
        for image in face_images:
            image_name = path_to_images + "/" + str(i) + ".jpg"
            i += 1
            # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(image_name, image)
        ExtractEmbeddings.main(self.dataset, self.embeddings, self.proto, self.model, self.embedding_model)
        TrainModel.main(self.embeddings, self.recognizer, self.le)

        path_to_user_images = self.dataset + "/" + user
        shutil.rmtree(path_to_user_images)
        # zip output so it can be sent easily
        zipname = "%s_frdata.zip" % user
        output_zip = zipfile.ZipFile(zipname, 'w')
        # for folder, subfolders, files in os.walk("output"):
        #     for file in files:
        #         output_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder, file), "output"),
        #                          compress_type=zipfile.ZIP_DEFLATED)
        output_zip.write("output/le.pickle", "le.pickle")
        output_zip.write("output/recognizer.pickle", "recognizer.pickle")

        output_zip.close()
        return zipname

# dataset = "temp/dataset"
# proto = "face_detection_model/deploy.prototxt"
# model = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
# embeddings = "output/embeddings.pickle"
# recognizer = "output/recognizer.pickle"
# le = "output/le.pickle"
# embedding_model = "face_detection_model/openface_nn4.small2.v1.t7"
#
#
# lf = LearnFace(dataset, proto, model, embeddings, recognizer, le, embedding_model)
# lf.learn_face(10, "toplexil40@gmail.com")
