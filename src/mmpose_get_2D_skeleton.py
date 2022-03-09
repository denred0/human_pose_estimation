import cv2
import os
import numpy as np
import pickle
import collections
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model,
                         vis_pose_result,
                         process_mmdet_results,
                         inference_pose_lifter_model)

from mmpose.core.visualization.image import imshow_keypoints

from mmpose.apis.inference import _xywh2xyxy

from tqdm import tqdm
from my_utils import recreate_folder, coco_to_h36m, get_all_files_in_folder
from constants import H36M_POSE_CONNECTIONS


def get_2D_keypoints(det_config: str,
                     det_checkpoint: str,
                     pose_config2d: str,
                     pose_checkpoint2d: str,
                     input_dir: str,
                     output_dir: str,
                     output_crops_dir: str,
                     save_results_txt: bool,
                     convert_coco_to_h36m: bool,
                     output_h36m_dir: str,
                     output_h36m_txt_dir: str) -> None:
    # initialize pose model
    pose_model2d = init_pose_model(pose_config2d, pose_checkpoint2d)

    # initialize detector
    det_model = init_detector(det_config, det_checkpoint)

    image_files = get_all_files_in_folder(input_dir, ["*.jpg"])

    result_keypoints = []
    for f in tqdm(image_files):

        mmdet_results = inference_detector(det_model, f)

        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        pose_results, returned_outputs = inference_top_down_pose_model(pose_model2d,
                                                                       f,
                                                                       person_results,
                                                                       bbox_thr=0.7,
                                                                       format='xyxy',
                                                                       dataset=pose_model2d.cfg.data.test.type)
        # remove face landmarks
        # try:
        #     if len(pose_results):
        #         for j in range(len(pose_results)):
        #             for i, keyp in enumerate(pose_results[j]['keypoints']):
        #                 if 23 <= i < 91 or i == 0 or i == 3 or i == 4:
        #                     pose_results[j]['keypoints'][i] = [0, 0, 0]
        # except:
        #     print()

        vis_result = vis_pose_result(pose_model2d,
                                     f,
                                     pose_results,
                                     dataset=pose_model2d.cfg.data.test.type,
                                     show=False,
                                     kpt_score_thr=0.0)

        cv2.imwrite(os.path.join(output_dir, f.name), vis_result)

        if convert_coco_to_h36m:
            h36m_keypoints = coco_to_h36m(pose_results[0]['keypoints'].tolist())

            h36m_vis = imshow_keypoints(cv2.imread(str(f)),
                                        [h36m_keypoints],
                                        pose_kpt_color=[[0, 255, 0]] * len(h36m_keypoints),
                                        pose_link_color=[[255, 0, 0]] * len(h36m_keypoints),
                                        skeleton=H36M_POSE_CONNECTIONS)
            vis_convert_result = cv2.hconcat([vis_result, h36m_vis])
            cv2.imwrite(os.path.join(output_h36m_dir, f.name), vis_convert_result)

            with open(os.path.join(output_h36m_txt_dir, str(f.stem) + ".txt"), 'w') as file:
                for idx, item in enumerate(h36m_keypoints):
                    file.write("%s\n" % (str(int(round(item[0], 0))) + " " + str(int(round(item[1], 0))) + " " +
                               str(round(item[2], 3))))

        if len(pose_results) > 0:
            box = pose_results[0]['bbox']

            # vis_result = vis_result[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            # cv2.imwrite(os.path.join(keypoint_inference, f.name), vis_result)

            img_orig = cv2.imread(str(f))
            img_orig = img_orig[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            cv2.imwrite(os.path.join(output_crops_dir, f.name), img_orig)

            keypoints = pose_results[0]['keypoints'].tolist()
            for i, k in enumerate(keypoints):
                x = int(k[0] - box[0])
                y = int(k[1] - box[1])

                result_keypoints.append(str(f.name) + " " + str(i) + " " + str(x) + " " + str(y))

        if save_results_txt:
            with open('data/mmpose_get_2D_skeleton/results_pytorch.txt', 'w') as file:
                for item in result_keypoints:
                    file.write("%s\n" % item)

if __name__ == "__main__":
    det_config = 'mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    det_checkpoint = 'pretrained_weights/mmdetection /cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

    pose_config2d = 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_384x288_udp.py'
    pose_checkpoint2d = 'pretrained_weights/mmpose/body/hrnet_w32_coco_384x288_udp-e97c1a0f_20210223.pth'

    input_dir = 'data/mmpose_get_2D_skeleton/input'
    output_dir = 'data/mmpose_get_2D_skeleton/output/images'
    recreate_folder(output_dir)

    output_crops_dir = 'data/mmpose_get_2D_skeleton/output/crops'
    recreate_folder(output_crops_dir)

    output_h36m_images_dir = 'data/mmpose_get_2D_skeleton/output/coco_to_h36m_images'
    recreate_folder(output_h36m_images_dir)
    output_h36m_txt_dir = 'data/mmpose_get_2D_skeleton/output/coco_to_h36m_txt'
    recreate_folder(output_h36m_txt_dir)

    save_results_txt = True
    convert_coco_to_h36m = True

    get_2D_keypoints(det_config,
                     det_checkpoint,
                     pose_config2d,
                     pose_checkpoint2d,
                     input_dir,
                     output_dir,
                     output_crops_dir,
                     save_results_txt,
                     convert_coco_to_h36m,
                     output_h36m_images_dir,
                     output_h36m_txt_dir)
