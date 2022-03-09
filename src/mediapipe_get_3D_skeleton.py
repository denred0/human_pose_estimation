import cv2
import os
import json
import math
import numpy as np
import mediapipe as mp

from tqdm import tqdm
from typing import List, Mapping, Optional, Tuple, Union

from my_utils import recreate_folder, get_all_files_in_folder

POSE_NAME = {0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
             4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 7: "left_ear",
             8: "right_ear", 9: "mouth_left", 10: "mouth_right", 11: "left_shoulder",
             12: "right_shoulder", 13: "left_elbow", 14: "right_elbow", 15: "left_wrist",
             16: "right_wrist", 17: "left_pinky", 18: "right_pinky", 19: "left_index",
             20: "right_index", 21: "left_thumb", 22: "right_thumb", 23: "left_hip",
             24: "right_hip", 25: "left_knee", 26: "right_knee", 27: "left_ankle",
             28: "right_ankle", 29: "left_heel", 30: "right_heel", 31: "left_foot_index",
             32: "right_foot_index"}

POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_ketpoints_from_images(input_dir: str,
                              ext: str,
                              output_dir: str,
                              enable_segmentation: bool,
                              min_detection_confidence: float,
                              save_json: bool) -> None:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    BG_COLOR = (192, 192, 192)  # gray

    images = get_all_files_in_folder(input_dir, [f"*.{ext}"])

    result = {}

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence) as pose:
        for im in tqdm(images):
            image = cv2.imread(str(im))
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue
            # print(
            #     f'Nose coordinates: ('
            #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            # )

            annotated_image = image.copy()
            if enable_segmentation:
                # Draw segmentation on the image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image".
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
                annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imwrite(os.path.join(output_dir, im.name), annotated_image)
            # Plot pose world landmarks.
            # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

            idx_to_coordinates = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # if ((landmark.HasField('visibility') and
                #      landmark.visibility < 0.5) or
                #         (landmark.HasField('presence') and
                #          landmark.presence < 0.5)):
                #     continue
                landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                               image_width, image_height)
                if landmark_px:
                    idx_to_coordinates[idx] = landmark_px

            result[im.name] = {"relative": results.pose_landmarks, "absolute": idx_to_coordinates}

        if save_json:
            save_keypoints_to_json(result)


def save_keypoints_to_json(keypoints_result):
    data = []

    for img_name, keypoints in tqdm(keypoints_result.items()):

        image_data = {}
        image_data["image"] = img_name

        keypoints_data = []
        keypoints_rel = keypoints["relative"]
        keypoints_abs = keypoints["absolute"]

        for idx, (landmark_rel, landmark_abs) in enumerate(zip((keypoints_rel.landmark), list(keypoints_abs.values()))):
            keypoints_data.append({"id": idx,
                                   "name": POSE_NAME[idx],
                                   "x_relative": landmark_rel.x,
                                   "y_relative": landmark_rel.y,
                                   "z_relative": landmark_rel.z,
                                   "x_absolute": landmark_abs[0],
                                   "y_absolute": landmark_abs[1],
                                   "visibility": float(str(round(landmark_rel.visibility, 2))), })

        image_data["keypoints"] = keypoints_data
        data.append(image_data)

    # with open("results_json.txt", 'w') as f:
    #     for item in data:
    #         f.write("%s\n" % str(item).replace("\'", "\""))

    with open('keypoints_result.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)


if __name__ == "__main__":
    input_dir = "data/mediapipe_get_skeleton/input"
    ext = "jpg"

    output_dir = "data/mediapipe_get_skeleton/output"
    recreate_folder(output_dir)

    enable_segmentation = False
    min_detection_confidence = 0.5
    save_json = True

    get_ketpoints_from_images(input_dir,
                              ext,
                              output_dir,
                              enable_segmentation,
                              min_detection_confidence,
                              save_json)
