#!/usr/bin/env python3

import argparse
import time
from typing import Optional, Tuple

import ctrlutils
import cv2
import numpy as np
import torch
import torch.backends.cudnn

import tools._init_paths
import fcn.config  # type: ignore
from lib.fcn import test_dataset
from lib import networks

print(f"Added {tools._init_paths.lib_path} to path.")

CFG = fcn.config.cfg


def load_networks(
    args: argparse.Namespace, num_classes: int = 2
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    network_data = torch.load(args.pretrained)
    print(f"=> using pre-trained network {args.pretrained}")
    network = networks.__dict__[args.network_name](
        num_classes, CFG.TRAIN.NUM_UNITS, network_data
    ).to(CFG.device)
    torch.backends.cudnn.benchmark = True
    network.eval()

    if args.refined:
        network_data_crop = torch.load(args.pretrained_crop)
        network_crop = networks.__dict__[args.network_name](
            num_classes, CFG.TRAIN.NUM_UNITS, network_data_crop
        ).to(CFG.device)
        network_crop.eval()
    else:
        network_crop = None

    return network, network_crop


def compute_xyz(
    depth_img: np.ndarray, depth_intrinsic: np.ndarray, height: int, width: int
) -> np.ndarray:
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    px = depth_intrinsic[0, 2]
    py = depth_intrinsic[1, 2]
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def segment(
    network: torch.nn.Module,
    network_refine: Optional[torch.nn.Module],
    img_color: np.ndarray,
    img_depth: np.ndarray,
    depth_intrinsic: np.ndarray,
) -> np.ndarray:
    img_color = img_color.astype(np.float32) / 255.0
    pixel_mean = torch.tensor(CFG.PIXEL_MEANS / 255.0).float()
    tensor_color = (torch.from_numpy(img_color) - pixel_mean).permute(2, 0, 1)
    img_xyz = compute_xyz(
        img_depth, depth_intrinsic, img_depth.shape[0], img_depth.shape[1]
    )
    tensor_xyz = torch.from_numpy(img_xyz).permute(2, 0, 1)

    sample = {
        "image_color": tensor_color.unsqueeze(0).to(CFG.device),
        "depth": tensor_xyz.unsqueeze(0).to(CFG.device),
    }

    out_label, out_label_refined = test_dataset.test_sample(
        sample, network, network_refine
    )

    if out_label_refined is not None:
        # Refined has finer-grained segmentation within objects.
        img_label = out_label_refined[0].cpu().numpy().astype(np.uint8)
    else:
        img_label = out_label[0].cpu().numpy().astype(np.uint8)

    return img_label


def match_instances(
    img_labels: np.ndarray, img_instances: Optional[np.ndarray]
) -> np.ndarray:
    if img_instances is None:
        return img_labels

    labels = [idx_label for idx_label in np.unique(img_labels) if idx_label != 0]
    img_instances_new = np.zeros_like(img_labels)
    idx_instance_next = img_instances.max() + 1
    for idx_label in labels:
        img_label = img_labels == idx_label
        img_instance = img_instances[img_label]

        # Find instance with most overlap with label.
        instance_labels, instance_counts = np.unique(img_instance, return_counts=True)
        idx_instance = instance_labels[instance_counts.argmax()]

        # Assign new instance idx if there is not enough overlap.
        if idx_instance == 0:
            idx_instance = idx_instance_next
            idx_instance_next += 1

        img_instances_new[img_label] = idx_instance

    return img_instances_new


def draw_labels(img_color: np.ndarray, img_labels: np.ndarray) -> np.ndarray:
    img = np.zeros_like(img_color)
    labels = [idx_label for idx_label in np.unique(img_labels) if idx_label != 0]
    for idx_label in labels:
        img_label = img_labels == idx_label

        # Overlay rgb image in segmentation.
        img[img_label, :] = img_color[img_label, :]

        # Find centroid of segmented object.
        moments = cv2.moments(img_label.astype(np.uint8))  # type: ignore
        xy_label = (np.array([moments["m10"], moments["m01"]]) / moments["m00"]).astype(
            int
        )

        # Label segmented object.
        cv2.putText(  # type: ignore
            img, str(idx_label), xy_label, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 8  # type: ignore
        )
        cv2.putText(  # type: ignore
            img, str(idx_label), xy_label, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2  # type: ignore
        )

    return img


def main(args: argparse.Namespace):
    fcn.config.cfg_from_file(args.cfg_file)
    CFG.gpu_id = 0
    CFG.device = torch.device(f"cuda:{CFG.gpu_id}")
    CFG.MODE = "TEST"

    redis = ctrlutils.RedisClient(args.redis_host, args.redis_port, args.redis_pass)
    b_depth_intrinsic = redis.get("rgbd::camera_0::depth::intrinsic")
    assert type(b_depth_intrinsic) is bytes
    depth_intrinsic = ctrlutils.redis.decode_matlab(b_depth_intrinsic)

    network, network_refine = load_networks(args)

    redis_pipe = redis.pipeline()
    img_instances = None

    while True:
        redis_pipe.get("rgbd::camera_0::color")
        redis_pipe.get("rgbd::camera_0::depth")
        b_img_color, b_img_depth = redis_pipe.execute()
        img_color = ctrlutils.redis.decode_opencv(b_img_color)
        img_depth = ctrlutils.redis.decode_opencv(b_img_depth) / 1000

        tic = time.time()
        img_labels = segment(
            network, network_refine, img_color, img_depth, depth_intrinsic
        )
        img_instances = match_instances(img_labels, img_instances)
        print(f"{time.time() - tic}s", len(np.unique(img_labels)) - 1, "objects")

        # cv2.imshow("color", img_color)  # type: ignore
        # cv2.imshow("depth", img_depth / img_depth.max())  # type: ignore
        cv2.imshow("label", draw_labels(img_color, img_instances))  # type: ignore
        key = cv2.waitKey(1)  # type: ignore
        if key >= 0 and chr(key) == "q":
            break

        redis.set_image("segmentation::label", img_instances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        default="experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml",
    )
    parser.add_argument(
        "--network", dest="network_name", default="seg_resnet34_8s_embedding"
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default="data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth",
    )
    parser.add_argument(
        "--pretrained_crop",
        dest="pretrained_crop",
        default="data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth",
    )
    parser.add_argument("--refined", default=False)
    parser.add_argument("-rh", dest="redis_host", default="127.0.0.1")
    parser.add_argument("-p", dest="redis_port", default=6379)
    parser.add_argument("-a", dest="redis_pass", default="")
    args = parser.parse_args()

    main(args)
