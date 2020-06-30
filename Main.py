import os
import cv2 as cv
import numpy as np

os.chdir('C:/Users/F112974/surfdrive/Onderzoek/AweSome/deduce_instagram_06_2020/code_blur')

import module_find_blur_faces as mfbf
import module_find_blur_text as mfbt

directory = r"C:\Users\F112974\surfdrive\Onderzoek\AweSome\deduce_instagram_06_2020\lauraboeschoten_20200420"

path_list_jpg = [os.path.join(dirpath, filename) for dirpath, _,
                                                     filenames in os.walk(directory) for filename in filenames if
                 filename.endswith('.jpg')]

for i in range(len(path_list_jpg)):

    # Blur faces on images
    img = cv.imread(path_list_jpg[i])
    frame_bf = mfbf.find_blur_faces(img)

    # Blur text on the images that already contain blurred faces
    frame_bt = mfbt.find_text_and_blur(
        frame_bf,
        eastPath = "C:/Users/F112974/surfdrive/Onderzoek/AweSome/tests/test_text_blur/frozen_east_text_detection.pb",
        min_confidence = 0.5)

    cv.imwrite(path_list_jpg[i], frame_bt)



# videos

path_list_mp4 = [os.path.join(dirpath, filename) for dirpath, _,
                                                     filenames in os.walk(directory) for filename in filenames if
                 filename.endswith('.mp4')]

for i in range(len(path_list_mp4)):

    cap = cv.VideoCapture(path_list_mp4[i])
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    img_array = []

    for g in range(total_frames):
        cap.set(1, g - 1);
        success = cap.grab()
        ret, image = cap.retrieve()

        #if ret == True:
        # blur faces on frame
        frame_bf = mfbf.find_blur_faces(image)
        # blur text on frame
        frame_bt = mfbt.find_text_and_blur(
            frame_bf,
            eastPath="C:/Users/F112974/surfdrive/Onderzoek/AweSome/tests/test_text_blur/frozen_east_text_detection.pb",
            min_confidence=0.5)

        img_array.append(frame_bt)

        #else:

    height, width, layers = image.shape
    size = (width, height)

    out = cv.VideoWriter(path_list_mp4[i][:-4] + '.mp4', cv.VideoWriter_fourcc(*'DIVX'), 15, size)

    # store the blurred video
    for f in range(len(img_array)):
        cvimage = np.array(img_array[f])
        out.write(cvimage)

    out.release()
