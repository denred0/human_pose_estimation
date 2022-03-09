from typing import List

import numpy as np

from tqdm import tqdm

from mmpose_get_2D_skeleton import get_2D_keypoints
from my_utils import recreate_folder, get_all_files_in_folder
from poserie.common.camera import normalize_screen_coordinates, image_coordinates
from poserie.common.generators import ChunkedGenerator, Evaluate_Generator
from pose_rie_run import evaluate


def create_3D_render(input_txt_dir: str, frame_size: List):
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

    gen = Evaluate_Generator(batch_size_param, None, None, [input_keypoints], stride_param,
                             pad=pad, causal_shift=causal_shift, augment=test_time_augmentation,
                             shuffle=False,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                             joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)

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
        # cam = dataset.cameras()[viz_subject_param][viz_camera_param]
        # if ground_truth is not None:
        #     prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
        #     ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        # else:
        #     # If the ground truth is not available, take the camera extrinsic params from a random subject.
        #     # They are almost the same, and anyway, we only need this for visualization purposes.
        #     for subject in dataset.cameras():
        #         if 'orientation' in dataset.cameras()[subject][viz_camera_param]:
        #             rot = dataset.cameras()[subject][viz_camera_param]['orientation']
        #             break
        #     prediction = camera_to_world(prediction, R=rot, t=0)
        #     # We don't have the trajectory, but at least we can rebase the height
        #     prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=frame_size[0], h=frame_size[1])

        from poserie.common.visualization import render_animation

        keypoints_metadata = {'num_joints': 17, 'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]}

        dataset_name = "h36m"
        dataset_path = 'poserie/data/data_3d_' + dataset_name + '.npz'
        if dataset_name == 'h36m':
            from poserie.common.h36m_dataset import Human36mDataset

            dataset = Human36mDataset(dataset_path)

        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), viz_bitrate, cam_azimuth, viz_output_param,
                         limit=viz_limit, downsample=viz_downsample, size=viz_size,
                         input_video_path=viz_video, viewport=(frame_size[0], frame_size[1]),
                         input_video_skip=viz_skip, viz_action=viz_action_param, viz_subject=viz_subject_param)

    print()


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

    frame_size = [1280, 720]
    create_3D_render(output_h36m_txt_dir, frame_size)
