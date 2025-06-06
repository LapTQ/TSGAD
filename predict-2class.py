from args import create_exp_dirs
from args import init_parser, init_sub_args
import torch
import random
import numpy as np
import lib.dist as dist
from lib.flows import FactorialNormalizingFlow
from dataset import get_dataset_and_loader
from utils.train_utils import (
    dump_args,
    init_model_params,
    Trainer,
    init_optimizer,
    init_scheduler,
)
from utils.data_utils import trans_list
from models import VAE
from tqdm import tqdm
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
import random
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json
import os
import time


LOGS = {
    "tsgad_infer_time": [],
}


def run(**kwargs):
    sequence_length = kwargs["sequence_length"]
    device = kwargs["device"]
    joint_order = kwargs["joint_order"]
    model_path = kwargs["model_path"]
    model_backend = kwargs["model_backend"]
    pathd_lbl = kwargs["pathd_lbl"]
    pathd_output = kwargs["pathd_output"]
    threshold = kwargs["threshold"]
    pathf_means = kwargs["pathf_means"]

    assert model_backend in ["torch", "tensorrt"]

    if pathd_output is not None:
        os.makedirs(pathd_output, exist_ok=True)

    class Args:
        pass

    fake_args = Args()
    fake_args.latent_dim = 10
    fake_args.exclude_mutinfo = False
    fake_args.tcvae = True
    fake_args.conv = False
    fake_args.graph = True
    fake_args.mss = False
    fake_args.dropout = 0.3
    fake_args.conv_oper = "sagc"
    fake_args.act = "relu"
    fake_args.headless = True
    fake_args.seg_len = sequence_length
    fake_args.mse = True
    fake_args.alpha = 1
    fake_args.gamma = 1

    prior_dist = dist.Normal()
    q_dist = dist.Normal()

    if model_backend == "torch":
        model = VAE(
            z_dim=fake_args.latent_dim,
            use_cuda=True,
            device=device,
            prior_dist=prior_dist,
            q_dist=q_dist,
            include_mutinfo=not fake_args.exclude_mutinfo,
            tcvae=fake_args.tcvae,
            conv=fake_args.conv,
            graph=fake_args.graph,
            mss=fake_args.mss,
            drop_out=fake_args.dropout,
            conv_oper=fake_args.conv_oper,
            act=fake_args.act,
            headless=fake_args.headless,
            input_frames=fake_args.seg_len,
            mse=fake_args.mse,
            alpha=fake_args.alpha,
            gamma=fake_args.gamma,
        )
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
        model.to(device)
        model.eval()
    elif model_backend == "tensorrt":
        raise NotImplementedError("TensorRT backend is not implemented yet.")

    # Load mean embeddings for both classes
    with open(pathf_means, "rb") as f:
        means_dict = pickle.load(f)
    mean_class1 = torch.from_numpy(means_dict["mean_class1"]).to(device)
    mean_class2 = torch.from_numpy(means_dict["mean_class2"]).to(device)

    info_tracks = {}
    for namef_lbl in tqdm(sorted(os.listdir(pathd_lbl))):
        pathf_lbl = os.path.join(pathd_lbl, namef_lbl)

        # load cached pose prediction from YOLO
        with open(pathf_lbl, "r") as f:
            dict_result = json.load(f)

        list__obj__box_xcycwhn = dict_result["list__obj__box_xcycwhn"]
        list__obj__kpts_xyn = dict_result["list__obj__kpts_xyn"]
        list__obj__id_track = dict_result.get(
            "list__obj__id_track", [None] * len(list__obj__box_xcycwhn)
        )

        dict_result["list__obj__action_conf"] = {
            "0": [],
            "1": [],
            "unk": [],
        }
        dict_result["list__obj__action_status"] = {
            "0": [],
            "1": [],
            "unk": [],
        }
        for i_obj, (box_xcycwhn, kpts_xyn, id_track) in enumerate(
            zip(list__obj__box_xcycwhn, list__obj__kpts_xyn, list__obj__id_track)
        ):

            # shift keypoints w.r.t bounding box
            b_xcn, b_ycn, b_wn, b_hn = box_xcycwhn
            b_x1n = b_xcn - b_wn / 2
            b_y1n = b_ycn - b_hn / 2
            b_x2n = b_x1n + b_wn
            b_y2n = b_y1n + b_hn

            if kpts_xyn is not None:
                kpts_xyn_shifted = []
                for kname in joint_order:
                    k_xn, k_yn = kpts_xyn[kname]
                    if not (k_xn == 0 and k_yn == 0):
                        k_xn = (k_xn - b_x1n) / b_wn
                        k_yn = (k_yn - b_y1n) / b_hn
                    kpts_xyn_shifted.append([k_xn, k_yn])
            else:
                kpts_xyn_shifted = None

            if id_track not in info_tracks:
                info_tracks[id_track] = {
                    "seq__kpts_xyn_shifted": [],
                }
            if kpts_xyn_shifted is not None:
                info_tracks[id_track]["seq__kpts_xyn_shifted"].append(kpts_xyn_shifted)

            if len(info_tracks[id_track]["seq__kpts_xyn_shifted"]) < sequence_length:
                dict_result["list__obj__action_conf"]["0"].append(None)
                dict_result["list__obj__action_conf"]["1"].append(None)
                dict_result["list__obj__action_conf"]["unk"].append(None)
                dict_result["list__obj__action_status"]["0"].append(False)
                dict_result["list__obj__action_status"]["1"].append(False)
                dict_result["list__obj__action_status"]["unk"].append(True)
                continue

            seq_kpts = info_tracks[id_track]["seq__kpts_xyn_shifted"][-sequence_length:]
            seq_kpts = np.array(seq_kpts)[np.newaxis, :, :, :]

            with torch.no_grad():
                # rescale to [-1, 1]
                seq_kpts = seq_kpts * 2 - 1
                pts1 = torch.tensor(seq_kpts, dtype=torch.float32).permute(0, 3, 1, 2)

                mtime_1 = time.time()
                if model_backend == "torch":
                    pts1 = pts1.to(device, non_blocking=True)
                    _, z_params, _ = model.encode_v2(pts1)
                    z_embed = z_params[:, :, 0].view(1, -1)  # flatten

                    # Compute L2 distances to both means
                    dist1 = torch.norm(z_embed - mean_class1, p=2, dim=1)
                    dist2 = torch.norm(z_embed - mean_class2, p=2, dim=1)
                    dists = torch.stack([dist1, dist2], dim=1)
                    preds = torch.softmax(-dists, dim=1)
                    prob = preds[0, 1].item()  # probability of shoplifting (class2)
                elif model_backend == "tensorrt":
                    raise NotImplementedError(
                        "TensorRT backend is not implemented yet."
                    )
                mtime_2 = time.time()
                LOGS["tsgad_infer_time"].append(mtime_2 - mtime_1)

            # print(prob)

            dict_result["list__obj__action_conf"]["0"].append(
                float(prob)
            )  # should be 1-prob, but now prob of shoplifting for visualization
            dict_result["list__obj__action_conf"]["1"].append(float(prob))
            dict_result["list__obj__action_conf"]["unk"].append(None)

            if prob >= threshold:
                dict_result["list__obj__action_status"]["0"].append(False)
                dict_result["list__obj__action_status"]["1"].append(True)
                dict_result["list__obj__action_status"]["unk"].append(False)
            # elif prob > 0.3:
            #     dict_result["list__obj__action_status"]['0'].append(False)
            #     dict_result["list__obj__action_status"]['1'].append(False)
            #     dict_result["list__obj__action_status"]['unk'].append(True)
            else:
                dict_result["list__obj__action_status"]["0"].append(True)
                dict_result["list__obj__action_status"]["1"].append(False)
                dict_result["list__obj__action_status"]["unk"].append(False)

            # print(
            #     "ST-GCN infer time: {} ({} FPS)".format(
            #         np.average(LOGS["stgcn_infer_time"]),
            #         1 / np.average(LOGS["stgcn_infer_time"]),
            #     )
            # )

        if pathd_output is not None:
            with open(os.path.join(pathd_output, namef_lbl), "w") as f:
                json.dump(dict_result, f, indent=4)


if __name__ == "__main__":

    for subpathf in [
        "Shoplifting/Shoplifting__30_.mp4",

        # "shoplifting-25min.mp4",
        # "satudora-1min.mp4",
        # "1568080723085_67014_fix.mkv",
    ]:

        kwargs = {
            "model_path": "/home/laptq/laptq-fs26-shoplifting-detection/runs/TSGAD-2class--TRAIN-mnit-roboflow-poselift/results/tsgad--last.pth",
            "model_backend": "torch",
            # "model_path": "/home/laptq/laptq-fs26-shoplifting-detection/outputs/convert-onnx-to-tensorrt/tsstg-hand-model-last.trt",
            # "model_backend": "tensorrt",

            "pathd_lbl": "/home/laptq/laptq-fs26-shoplifting-detection/outputs/sample_frames_by_skipping/full/{}/labels--PRED--DATA--None--MODEL--yolov8x-pose--TRAIN--exp--PREDICT--imgsz-640--conf-0.4--iou-0.45--filterby-size--all-keypoints--JSON".format(
                subpathf
            ),
            # "pathd_lbl": "/home/laptq/laptq-fs26-shoplifting-detection/outputs/sample_frames_by_skipping/full/{}/labels--PRED--DATA--None--MODEL--yolov8x-pose--TRAIN--exp--PREDICT--imgsz-640--conf-0.4--iou-0.45--filterby-size--all-keypoints--JSON".format(
            #     subpathf
            # ),
            # "pathd_lbl": "/home/laptq/laptq-fs26-shoplifting-detection/outputs/sample_frames_by_skipping/full/{}/labels--PRED--DATA--None--MODEL--yolov8x-pose--TRAIN--exp--PREDICT--imgsz-640--conf-0.1--iou-0.45--all-keypoints--JSON".format(
            #     subpathf
            # ),
            # "pathd_lbl": "/home/laptq/laptq-fs26-shoplifting-detection/outputs/helper--extract--ultralytics--imgdir/{}/labels--PRED--DATA--None--MODEL--yolov8x-pose--TRAIN--exp--PREDICT--imgsz-640--conf-0.1--iou-0.45--all-keypoints--JSON".format(
            #     subpathf
            # ),

            "pathd_output": "/home/laptq/laptq-fs26-shoplifting-detection/outputs/TSGAD-2class--TRAIN-mnit-roboflow-poselift/predict/{}/labels".format(
                subpathf
            ),

            "sequence_length": 24,
            "device": "cuda:1",
            "threshold": 0.5,
            "joint_order": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
            "pathf_means": "/home/laptq/laptq-fs26-shoplifting-detection/runs/TSGAD-2class--TRAIN-mnit-roboflow-poselift/results/means_val.pkl",
        }
        run(**kwargs)
