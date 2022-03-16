from typing import List

import numpy as np
import torch
import os
import cv2
from tqdm import tqdm

from mmpose_get_2D_skeleton import get_2D_keypoints
from my_utils import recreate_folder, get_all_files_in_folder
from poserie.common.camera import normalize_screen_coordinates, image_coordinates, camera_to_world
from poserie.common.generators import ChunkedGenerator, Evaluate_Generator
from pose_rie_run import evaluate, fetch
from poserie.common.model import RIEModel


def create_3D_render(input_txt_dir: str,
                     frame_size: tuple,
                     batch_size_param,
                     stride_param,
                     pad,
                     causal_shift,
                     test_time_augmentation,
                     kps_left,
                     kps_right,
                     joints_left,
                     joints_right,
                     viz_export,
                     viz_output_param,
                     viz_no_ground_truth,
                     viz_bitrate,
                     cam_azimuth,
                     viz_downsample,
                     viz_limit,
                     viz_video,
                     viz_skip,
                     viz_action_param,
                     viz_subject_param,
                     viz_size,
                     architecture,
                     causal,
                     dropout,
                     channels,
                     latent_features_dim,
                     dense,
                     stage_param,
                     viz_camera_param,
                     resume,
                     finetune,
                     evaluate_param,
                     checkpoint_dir,
                     dataset_name,
                     fps):
    #
    txts = get_all_files_in_folder(input_txt_dir, ["*.txt"])

    keypoints = np.zeros([len(txts), 17, 2])
    for id, txt in enumerate(tqdm(txts, desc="Reading txts...")):
        with open(txt) as file:
            lines = file.readlines()
            lines = [d.replace("\n", "") for d in lines]
            lines = [d.split() for d in lines]
            lines = np.array([[int(d[0]), int(d[1])] for d in lines])
        keypoints[id] = lines

    keypoints[..., :2] = normalize_screen_coordinates(keypoints[..., :2], w=frame_size[0], h=frame_size[1])

    print('Rendering...')
    input_keypoints = keypoints.copy()
    ground_truth = None
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    filter_widths = [int(x) for x in architecture.split(',')]

    keypoints_metadata = {'num_joints': 17, 'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]}

    dataset_path = 'poserie/data/data_3d_' + dataset_name + '.npz'
    if dataset_name == 'h36m':
        from poserie.common.h36m_dataset import Human36mDataset

        dataset = Human36mDataset(dataset_path)

    # action_filter = None
    # subjects_test = "S9"
    # cameras_valid, poses_valid, poses_valid_2d = fetch(keypoints, dataset, subjects_test, action_filter, parse_3d_poses=False)

    model_pos = RIEModel(keypoints[0].shape[-2], keypoints[0].shape[-1], dataset.skeleton().num_joints(),
                         filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                         latten_features=latent_features_dim, dense=dense, is_train=False, Optimize1f=True,
                         stage=stage_param)

    gen = Evaluate_Generator(batch_size_param, None, None, [input_keypoints], stride_param,
                             pad=pad, causal_shift=causal_shift, augment=test_time_augmentation,
                             shuffle=False,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                             joints_right=joints_right)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    if resume or evaluate_param or finetune:
        filename = ""
        if resume != "":
            filename = resume
        elif evaluate_param != "":
            filename = evaluate_param
        else:
            filename = finetune
        chk_filename = os.path.join(checkpoint_dir, filename)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        # model_pos_train.load_state_dict(checkpoint['model_pos'])
        model_pos.load_state_dict(checkpoint['model_pos'])

    prediction = evaluate(model_pos, gen, joints_left, joints_right, return_predictions=True)

    if viz_export is not None:
        print('Exporting joint positions to', viz_export)
        # Predictions are in camera space
        np.save(viz_export, prediction)

    if viz_output_param is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[viz_subject_param][viz_camera_param]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][1]:
                    rot = dataset.cameras()[subject][viz_camera_param]['orientation']
                    break
            t = (0, 0, -1000)
            # t = (0, 0, 0)
            prediction = camera_to_world(prediction, R=rot, t=t)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=frame_size[0], h=frame_size[1])

        from poserie.common.visualization import render_animation


        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), fps, viz_bitrate, cam_azimuth, viz_output_param,
                         limit=viz_limit, downsample=viz_downsample, size=viz_size,
                         input_video_path=viz_video, viewport=(frame_size[0], frame_size[1]),
                         input_video_skip=viz_skip, viz_action="", viz_subject="")


if __name__ == "__main__":

    output_h36m_txt_dir = 'data/mmpose_get_2D_skeleton/output/coco_to_h36m_txt'
    output_h36m_images_dir = 'data/mmpose_get_2D_skeleton/output/coco_to_h36m_images'
    is_2d_exist = False

    if not is_2d_exist:
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

    images = get_all_files_in_folder(output_h36m_images_dir, ["*.jpg"])

    batch_size_param = 1024
    stride_param = 1
    pad = 121
    causal_shift = 0
    test_time_augmentation = True
    kps_left = [4, 5, 6, 11, 12, 13]
    kps_right = [1, 2, 3, 14, 15, 16]
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    viz_export = None
    viz_output_param = "result.mp4"
    viz_no_ground_truth = False
    viz_bitrate = 3000
    cam_azimuth = np.array(70., dtype=np.float32)
    viz_downsample = 1
    viz_limit = -1
    viz_video = None
    viz_skip = 0
    viz_action_param = "Waiting"
    viz_subject_param = "S9"
    viz_size = 5
    architecture = "3,3,3,3,3"
    causal = False
    dropout = 0.2
    channels = 256
    latent_features_dim = 256
    dense = False
    stage_param = 3
    viz_camera_param = 0
    resume = ""
    finetune = ""
    evaluate_param = "gt_pretrained.bin"
    checkpoint_dir = "poserie/checkpoint"
    dataset_name = "h36m"
    fps = 30

    if images:
        temp_img = cv2.imread(str(images[0]))
        frame_size = (temp_img.shape[1] // 2, temp_img.shape[0])
        create_3D_render(output_h36m_txt_dir,
                         frame_size,
                         batch_size_param,
                         stride_param,
                         pad,
                         causal_shift,
                         test_time_augmentation,
                         kps_left,
                         kps_right,
                         joints_left,
                         joints_right,
                         viz_export,
                         viz_output_param,
                         viz_no_ground_truth,
                         viz_bitrate,
                         cam_azimuth,
                         viz_downsample,
                         viz_limit,
                         viz_video,
                         viz_skip,
                         viz_action_param,
                         viz_subject_param,
                         viz_size,
                         architecture,
                         causal,
                         dropout,
                         channels,
                         latent_features_dim,
                         dense,
                         stage_param,
                         viz_camera_param,
                         resume,
                         finetune,
                         evaluate_param,
                         checkpoint_dir,
                         dataset_name,
                         fps)
    else:
        print("fThere is no images in {output_h36m_images_dir} folder")
