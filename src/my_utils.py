import os
import shutil
import cv2
from tqdm import tqdm

from typing import List
from pathlib import Path


def get_all_files_in_folder(folder: str, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(Path(folder).rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def recreate_folders(root_dir: Path, folders_list: List) -> None:
    for directory in folders_list:
        output_dir = root_dir.joinpath(directory)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def recreate_folder(root_dir: str) -> None:
    output_dir = Path(root_dir)
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def BrightnessAndContrastAuto(im_source, clipHistPercent=0):
    histSize = 256
    minGray = 0
    maxGray = 0

    imgray = im_source.copy()
    if len(im_source.shape) == 3:
        imgray = cv2.cvtColor(im_source, cv2.COLOR_BGR2GRAY)

    if clipHistPercent == 0:
        # keep full available range
        minGray, maxGray, min_pt, max_pt = cv2.minMaxLoc(imgray)
    else:
        histRange = [0, 256]
        uniform = True
        accumulate = False
        hist = cv2.calcHist([imgray], [0], None, [histSize], histRange, uniform, accumulate)
        print()
        accumulator = [0] * len(hist)
        accumulator[0] = hist[0]
        for i in range(1, histSize):
            accumulator[i] = accumulator[i - 1] + hist[i]

        # locate points that cuts at required value
        max = accumulator[-1]
        clipHistPercent *= (max / 100.0)
        clipHistPercent /= 2.0
        minGray = 0

        #  locate left cut
        while accumulator[minGray] < clipHistPercent:
            minGray += 1

        # locate right cut
        maxGray = histSize - 1
        while accumulator[maxGray] >= (max - clipHistPercent):
            maxGray -= 1

    # current range
    inputRange = maxGray - minGray
    alpha = (histSize - 1) / inputRange
    beta = -minGray * alpha

    dst = imgray.copy()
    cv2.convertScaleAbs(imgray, dst, alpha, beta)

    return dst


def improve_brightness():
    output_foldet = "data/improve_brightness/output"
    recreate_folder(output_foldet)

    images = get_all_files_in_folder(Path("data/improve_brightness/input"), ["*.jpg"])

    for im in tqdm(images):
        img = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE)
        img_edit = BrightnessAndContrastAuto(img)
        cv2.imwrite(os.path.join(output_foldet, im.name), img_edit)


def generate_train_test(data_dir, split):
    images = get_all_files_in_folder(data_dir, ['*.png'])

    with open('data/create_txt_lists/' + split + '.txt', "w") as outfile:
        for image in images:
            outfile.write(split + '/' + image.name)
            outfile.write("\n")
        outfile.close()


def improve_brightness2():
    output_foldet = "data/improve_brightness/output"
    recreate_folder(output_foldet)

    images = get_all_files_in_folder(Path("data/improve_brightness/input"), ["*.jpg"])

    for im in tqdm(images):
        img = cv2.imread(str(im), cv2.IMREAD_COLOR)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels

        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv2.merge((l2, a, b))  # merge channels
        img_edit = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

        cv2.imwrite(os.path.join(output_foldet, im.name), img_edit)


def plot_one_box(im, box, label=None, color=(255, 255, 0), line_thickness=1):
    c1 = (box[0], box[1])
    c2 = (box[2], box[3])

    tl = line_thickness or round(0.001 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    im = cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        im = cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        im = cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return im


def coco_to_h36m(coco_keypoints: List) -> []:
    h36m_keypoints = [[0.0, 0.0, 0.0]] * len(coco_keypoints)

    h36m_keypoints[0] = [
        (max(coco_keypoints[11][0], coco_keypoints[12][0]) - min(coco_keypoints[11][0],
                                                                 coco_keypoints[12][0])) / 2 + min(
            coco_keypoints[11][0], coco_keypoints[12][0]), (
                max(coco_keypoints[11][1], coco_keypoints[12][1]) - min(coco_keypoints[11][1],
                                                                        coco_keypoints[12][1])) / 2 + min(
            coco_keypoints[11][1], coco_keypoints[12][1]), (coco_keypoints[11][2] + coco_keypoints[12][2]) / 2]
    h36m_keypoints[1] = coco_keypoints[11]
    h36m_keypoints[2] = coco_keypoints[13]
    h36m_keypoints[3] = coco_keypoints[15]
    h36m_keypoints[4] = coco_keypoints[12]
    h36m_keypoints[5] = coco_keypoints[14]
    h36m_keypoints[6] = coco_keypoints[16]
    h36m_keypoints[8] = [
        (max(coco_keypoints[5][0], coco_keypoints[6][0]) - min(coco_keypoints[5][0],
                                                               coco_keypoints[6][0])) / 2 + min(
            coco_keypoints[5][0], coco_keypoints[6][0]), (
                max(coco_keypoints[5][1], coco_keypoints[6][1]) - min(coco_keypoints[5][1],
                                                                      coco_keypoints[6][1])) / 2 + min(
            coco_keypoints[5][1], coco_keypoints[6][1]), (coco_keypoints[5][2] + coco_keypoints[6][2]) / 2]
    h36m_keypoints[9] = coco_keypoints[0]

    h36m_keypoints[11] = coco_keypoints[6]
    h36m_keypoints[12] = coco_keypoints[8]
    h36m_keypoints[13] = coco_keypoints[10]
    h36m_keypoints[14] = coco_keypoints[5]
    h36m_keypoints[15] = coco_keypoints[7]
    h36m_keypoints[16] = coco_keypoints[9]
    h36m_keypoints[7] = [
        (max(h36m_keypoints[0][0], h36m_keypoints[8][0]) - min(h36m_keypoints[0][0],
                                                               h36m_keypoints[8][0])) / 2 + min(
            h36m_keypoints[0][0], h36m_keypoints[8][0]), (
                max(h36m_keypoints[0][1], h36m_keypoints[8][1]) - min(h36m_keypoints[0][1],
                                                                      h36m_keypoints[8][1])) / 2 + min(
            h36m_keypoints[0][1], h36m_keypoints[8][1]), (h36m_keypoints[0][2] + h36m_keypoints[8][2]) / 2]

    if h36m_keypoints[8][1] > h36m_keypoints[9][1]:
        h36m_keypoints[10] = [
            (max(h36m_keypoints[8][0], h36m_keypoints[9][0]) - min(h36m_keypoints[8][0], h36m_keypoints[9][0])) / 2 +
            min(h36m_keypoints[8][0], h36m_keypoints[9][0]),
            h36m_keypoints[9][1] - (h36m_keypoints[8][1] - h36m_keypoints[9][1]) * 0.7, h36m_keypoints[9][2]]
    else:
        h36m_keypoints[10] = [0, 0, 0]

    return h36m_keypoints


if __name__ == "__main__":
    improve_brightness2()
