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
    evaluate_model,
)
from utils.data_utils import trans_list
from models import VAE
from tqdm import tqdm
from utils.eval import score_dataset, get_train_dist
import yaml
import os
from scipy.stats import norm, multivariate_normal
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pprint import pprint

# laptq
num_class = 2
SEED = 42


def load_dataset(data_files, batch_size, shuffle, split_size=0, sample_balance=False):
    """Load data files into torch DataLoader with/without spliting train-test."""
    features, labels = [], []
    for fil in data_files:
        if not os.path.exists(fil):
            print(f"File not found: {fil}")
            continue
        with open(fil, "rb") as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    print(features.shape)
    print(labels.shape)
    print(
        "Number of examples for each class:",
        [int(np.sum(labels[:, i])) for i in range(num_class)],
    )

    # sample balance
    if sample_balance:
        random.seed(SEED)
        min_num_samples = min(
            [
                int(np.sum(labels[:, i])) for i in range(num_class)
            ]  # assuming one-hot encoding
        )
        features_balanced = []
        labels_balanced = []
        for i in range(num_class):
            idxs = np.where(labels[:, i] == 1)[0]
            idxs = random.sample(list(idxs), k=min_num_samples)
            features_balanced.append(features[idxs])
            labels_balanced.append(labels[idxs])
        features = np.concatenate(features_balanced, axis=0)
        labels = np.concatenate(labels_balanced, axis=0)

        print("After sampling:")
        print(features.shape)
        print(labels.shape)

    print(
        "Number of examples for each class:",
        [int(np.sum(labels[:, i])) for i in range(num_class)],
    )

    # rescale to [-1, 1]
    features[..., :2] = 2 * features[..., :2] - 1
    labels = (labels[:, 1] == 1).astype(np.float32)  # Convert to binary labels (0 or 1)

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(
            features, labels, test_size=split_size, random_state=9
        )
        train_set = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
            torch.tensor(y_train, dtype=torch.float32),
        )
        valid_set = TensorDataset(
            torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
            torch.tensor(y_valid, dtype=torch.float32),
        )
        train_loader = DataLoader(train_set, batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_set, batch_size)
    else:
        train_set = TensorDataset(
            torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
            torch.tensor(labels, dtype=torch.float32),
        )
        train_loader = DataLoader(train_set, batch_size, shuffle=shuffle)
        valid_loader = None
    return train_loader, valid_loader


TRAIN_FILES = (
    [
        "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/mnit/Shoplifting/"
        + _
        for _ in """Shoplifting__10_.mp4.pkl                                                                                                                                                  
Shoplifting__11_.mp4.pkl                                                                                                                                                  
Shoplifting__12_.mp4.pkl                                                                                                                                                  
Shoplifting__13_.mp4.pkl                                                                                                                                                  
Shoplifting__14_.mp4.pkl                                                                                                                                                  
Shoplifting__15_.mp4.pkl                                                                                                                                                  
Shoplifting__16_.mp4.pkl                                                                                                                                                  
Shoplifting__17_.mp4.pkl                                                                                                                                                  
Shoplifting__18_.mp4.pkl                                                                                                                                                  
Shoplifting__19_.mp4.pkl                                                                                                                                                  
Shoplifting__1_.mp4.pkl                                                                                                                                                   
Shoplifting__20_.mp4.pkl                                                                                                                                                  
Shoplifting__21_.mp4.pkl                                                                                                                                                  
Shoplifting__22_.mp4.pkl                                                                                                                                                  
Shoplifting__23_.mp4.pkl                                                                                                                                                  
Shoplifting__24_.mp4.pkl                                                                                                                                                  
Shoplifting__25_.mp4.pkl                                                                                                                                                  
Shoplifting__26_.mp4.pkl                                                                                                                                                  
Shoplifting__27_.mp4.pkl                                                                                                                                                  
Shoplifting__28_.mp4.pkl                                                                                                                                                  
Shoplifting__29_.mp4.pkl                                                                                                                                                  
Shoplifting__2_.mp4.pkl                                                                                                                                                   
Shoplifting__30_.mp4.pkl                                                                                                                                                  
Shoplifting__31_.mp4.pkl                                                                                                                                                  
Shoplifting__32_.mp4.pkl                                                                                                                                                  
Shoplifting__33_.mp4.pkl                                                                                                                                                  
Shoplifting__34_.mp4.pkl                                                                                                                                                  
Shoplifting__35_.mp4.pkl                                                                                                                                                  
Shoplifting__36_.mp4.pkl                                                                                                                                                  
Shoplifting__37_.mp4.pkl                                                                                                                                                  
Shoplifting__38_.mp4.pkl                                                                                                                                                  
Shoplifting__39_.mp4.pkl                                                                                                                                                  
Shoplifting__3_.mp4.pkl                                                                                                                                                   
Shoplifting__40_.mp4.pkl                                                                                                                                                  
Shoplifting__41_.mp4.pkl                                                                                                                                                  
Shoplifting__42_.mp4.pkl                                                                                                                                                  
Shoplifting__43_.mp4.pkl                                                                                                                                                  
Shoplifting__44_.mp4.pkl                                                                                                                                                  
Shoplifting__45_.mp4.pkl                                                                                                                                                  
Shoplifting__46_.mp4.pkl                                                                                                                                                  
Shoplifting__47_.mp4.pkl
Shoplifting__48_.mp4.pkl
Shoplifting__4_.mp4.pkl
Shoplifting__50_.mp4.pkl
Shoplifting__51_.mp4.pkl
Shoplifting__52_.mp4.pkl
Shoplifting__53_.mp4.pkl
Shoplifting__54_.mp4.pkl
Shoplifting__55_.mp4.pkl
Shoplifting__56_.mp4.pkl
Shoplifting__57_.mp4.pkl
Shoplifting__58_.mp4.pkl
Shoplifting__59_.mp4.pkl
Shoplifting__5_.mp4.pkl
Shoplifting__60_.mp4.pkl
Shoplifting__61_.mp4.pkl
Shoplifting__62_.mp4.pkl
Shoplifting__63_.mp4.pkl
Shoplifting__64_.mp4.pkl
Shoplifting__65_.mp4.pkl
Shoplifting__66_.mp4.pkl""".split()
    ]
    + [
        "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/mnit/Normal/"
        + _
        for _ in """Normal__10_.mp4.pkl
Normal__11_.mp4.pkl
Normal__12_.mp4.pkl
Normal__13_.mp4.pkl
Normal__14_.mp4.pkl
Normal__15_.mp4.pkl
Normal__17_.mp4.pkl
Normal__19_.mp4.pkl
Normal__1_.mp4.pkl
Normal__20_.mp4.pkl
Normal__21_.mp4.pkl
Normal__22_.mp4.pkl
Normal__23_.mp4.pkl
Normal__25_.mp4.pkl
Normal__26_.mp4.pkl
Normal__27_.mp4.pkl
Normal__28_.mp4.pkl
Normal__29_.mp4.pkl
Normal__2_.mp4.pkl
Normal__30_.mp4.pkl
Normal__31_.mp4.pkl
Normal__32_.mp4.pkl
Normal__33_.mp4.pkl
Normal__34_.mp4.pkl
Normal__35_.mp4.pkl
Normal__36_.mp4.pkl
Normal__37_.mp4.pkl
Normal__38_.mp4.pkl
Normal__39_.mp4.pkl
Normal__3_.mp4.pkl
Normal__40_.mp4.pkl
Normal__41_.mp4.pkl
Normal__42_.mp4.pkl
Normal__43_.mp4.pkl
Normal__44_.mp4.pkl
Normal__46_.mp4.pkl
Normal__4_.mp4.pkl
Normal__52_.mp4.pkl
Normal__53_.mp4.pkl
Normal__54_.mp4.pkl
Normal__55_.mp4.pkl
Normal__56_.mp4.pkl
Normal__57_.mp4.pkl
Normal__58_.mp4.pkl
Normal__59_.mp4.pkl
Normal__5_.mp4.pkl
Normal__60_.mp4.pkl
Normal__61_.mp4.pkl
Normal__62_.mp4.pkl
Normal__63_.mp4.pkl
Normal__64_.mp4.pkl
Normal__65_.mp4.pkl""".split()
    ]
    #     # incorrect normal videos
    #     + [
    #         "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/mnit/Normal/"
    #         + _
    #         for _ in """Normal__16_.mp4.pkl
    # Normal__18_.mp4.pkl
    # Normal__24_.mp4.pkl
    # Normal__24_.mp4.pkl
    # Normal__45_.mp4.pkl
    # Normal__47_.mp4.pkl
    # Normal__48_.mp4.pkl
    # Normal__49_.mp4.pkl
    # Normal__50_.mp4.pkl
    # Normal__51_.mp4.pkl""".split()
    #     ]
    #     # poselift
    + [
        "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/poselift/"
        + _
        for _ in """test/01_0222/3.pkl
test/01_0225/1.pkl
test/01_0225/2.pkl
test/01_0225/9.pkl
test/01_0240/1.pkl
test/01_0242/1.pkl
test/01_0242/4.pkl
test/01_0245/1.pkl
test/01_0251/1.pkl
test/01_0251/2.pkl
test/01_0257/1.pkl
test/01_0272/1.pkl
test/01_0272/3.pkl
test/01_0273/1.pkl
test/01_0282/1.pkl
test/01_0282/2.pkl
test/01_0293/1.pkl
test/01_0293/2.pkl
test/02_0216/1.pkl
test/02_0270/1.pkl
test/02_0270/2.pkl
test/02_0290/1.pkl
test/02_0305/1.pkl
test/02_0310/2.pkl
test/02_0311/1.pkl
test/02_0313/1.pkl
test/02_0314/1.pkl
test/02_0316/1.pkl
test/02_0322/1.pkl
test/02_0322/3.pkl
test/02_0325/1.pkl
test/03_0219/1.pkl
test/03_0247/8.pkl
test/03_0248/1.pkl
test/03_0260/1.pkl
test/03_0260/2.pkl
test/03_0262/1.pkl
test/03_0262/4.pkl
test/03_0265/1.pkl
test/03_0265/2.pkl
test/03_0267/1.pkl
test/03_0267/2.pkl
test/03_0267/18.pkl
test/03_0276/1.pkl
test/03_0285/1.pkl
test/03_0287/2.pkl""".split()
    ]
    #     roboflow
    + [
        "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/roboflow/"
        + _
        for _ in """0.pkl
2.pkl
4.pkl
5.pkl
6.pkl
7.pkl
8.pkl
9.pkl
10.pkl
11.pkl
12.pkl
13.pkl
16.pkl
17.pkl
18.pkl
18_2.pkl
22.pkl
23.pkl
26.pkl
28.pkl
29.pkl""".split()
    ]
)

VAL_FILES = (
    [
        "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/mnit/Shoplifting/"
        + _
        for _ in """Shoplifting__67_.mp4.pkl                                                                                                                                                  
Shoplifting__68_.mp4.pkl                                                                                                                                                  
Shoplifting__69_.mp4.pkl                                                                                                                                                  
Shoplifting__6_.mp4.pkl                                                                                                                                                   
Shoplifting__70_.mp4.pkl                                                                                                                                                  
Shoplifting__71_.mp4.pkl                                                                                                                                                  
Shoplifting__72_.mp4.pkl                                                                                                                                                  
Shoplifting__73_.mp4.pkl                                                                                                                                                  
Shoplifting__74_.mp4.pkl                                                                                                                                                  
Shoplifting__75_.mp4.pkl                                                                                                                                                  
Shoplifting__76_.mp4.pkl                                                                                                                                                  
Shoplifting__77_.mp4.pkl                                                                                                                                                  
Shoplifting__78_.mp4.pkl
Shoplifting__79_.mp4.pkl
Shoplifting__7_.mp4.pkl
Shoplifting__80_.mp4.pkl
Shoplifting__81_.mp4.pkl
Shoplifting__82_.mp4.pkl
Shoplifting__83_.mp4.pkl
Shoplifting__84_.mp4.pkl
Shoplifting__85_.mp4.pkl
Shoplifting__86_.mp4.pkl
Shoplifting__87_.mp4.pkl
Shoplifting__88_.mp4.pkl
Shoplifting__89_.mp4.pkl
Shoplifting__8_.mp4.pkl
Shoplifting__90_.mp4.pkl
Shoplifting__91_.mp4.pkl
Shoplifting__92_.mp4.pkl
Shoplifting__93_.mp4.pkl
Shoplifting__9_.mp4.pkl""".split()
    ]
    + [
        "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/mnit/Normal/"
        + _
        for _ in """Normal__66_.mp4.pkl
Normal__67_.mp4.pkl
Normal__69_.mp4.pkl
Normal__6_.mp4.pkl
Normal__70_.mp4.pkl
Normal__71_.mp4.pkl
Normal__72_.mp4.pkl
Normal__73_.mp4.pkl
Normal__74_.mp4.pkl
Normal__75_.mp4.pkl
Normal__76_.mp4.pkl
Normal__77_.mp4.pkl
Normal__78_.mp4.pkl
Normal__79_.mp4.pkl
Normal__7_.mp4.pkl
Normal__80_.mp4.pkl
Normal__81_.mp4.pkl
Normal__82_.mp4.pkl
Normal__83_.mp4.pkl
Normal__84_.mp4.pkl
Normal__85_.mp4.pkl
Normal__86_.mp4.pkl
Normal__87_.mp4.pkl
Normal__88_.mp4.pkl
Normal__89_.mp4.pkl
Normal__8_.mp4.pkl
Normal__90_.mp4.pkl
Normal__9_.mp4.pkl""".split()
    ]
    # # incorrect normal videos
    # + [
    #     "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/mnit/Normal/"
    #     + _
    #     for _ in """Normal__68_.mp4.pkl""".split()
    # ]
    # poselift
    + [
        "/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/poselift/"
        + _
        for _ in """test/03_0296/1.pkl
test/04_0213/1.pkl
test/04_0234/1.pkl
test/04_0234/2.pkl
test/04_0234/3.pkl
test/04_0234/4.pkl
test/04_0237/1.pkl
test/04_0301/2.pkl
test/04_0302/1.pkl
test/04_0308/1.pkl
test/04_0319/1.pkl
test/05_0228/1.pkl
test/05_0228/2.pkl
test/05_0231/1.pkl
test/05_0254/2.pkl
test/05_0298/1.pkl
test/05_0299/1.pkl
test/06_0112/19.pkl
test/06_0279/3.pkl
test/06_0279/16.pkl""".split()
    ]
)


def main():
    print("sedaye mano darid az Chalotte America!")
    parser = init_parser()
    args = parser.parse_args()

    args.kp18_format = eval(args.kp18_format) if args.kp18_format is not None else None

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)
    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained_model = vars(args).get("model_ckpt_dir", None)

    # if not os.path.exists(
    #     "/home/laptq/laptq-fs26-shoplifting-detection/outputs/trivials/tsgad-data.pkl"
    # ):
    #     dataset, loader = get_dataset_and_loader(
    #         args, trans_list=trans_list, only_test=(pretrained_model is not None)
    #     )
    #     with open(
    #         "/home/laptq/laptq-fs26-shoplifting-detection/outputs/trivials/tsgad-data.pkl",
    #         "wb",
    #     ) as f:
    #         pickle.dump((dataset, loader), f)
    # else:
    #     with open(
    #         "/home/laptq/laptq-fs26-shoplifting-detection/outputs/trivials/tsgad-data.pkl",
    #         "rb",
    #     ) as f:
    #         dataset, loader = pickle.load(f)
    dataset, loader = get_dataset_and_loader(
        args, trans_list=trans_list, only_test=(pretrained_model is not None)
    )

    # model_args = init_model_params(args, dataset)

    # laptq
    train_2ndloader, _ = load_dataset(
        TRAIN_FILES,
        64,
        shuffle=True,
        split_size=0,
        sample_balance=False,
    )

    val_loader, _ = load_dataset(
        VAL_FILES,
        64,
        shuffle=False,
        split_size=0,
        sample_balance=False,
    )

    prior_dist = dist.Normal()
    q_dist = dist.Normal()

    vae = VAE(
        z_dim=model_args.latent_dim,
        use_cuda=True,
        device=args.device,
        prior_dist=prior_dist,
        q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo,
        tcvae=args.tcvae,
        conv=args.conv,
        graph=args.graph,
        mss=args.mss,
        drop_out=args.dropout,
        conv_oper=args.conv_oper,
        act=args.act,
        headless=args.headless,
        input_frames=args.seg_len,
        mse=args.mse,
        alpha=args.model_alpha,
        gamma=args.model_gamma,
    )

    assert args.task in ["train", "val"]

    if args.task == "train":
        if not os.path.exists(args.model_save_dir):
            # Create the directory
            os.makedirs(args.model_save_dir)
        arguments = vars(args)
        with open(args.model_save_dir + "/" + "arguments.yaml", "w") as file:
            yaml.dump(arguments, file)
        ae_optimizer_f = init_optimizer(args.model_optimizer, lr=args.model_lr)
        ae_scheduler_f = init_scheduler(
            args.sched, lr=args.model_lr, epochs=args.epochs
        )
        trainer = Trainer(
            model_args,
            vae,
            loader["train"],
            loader["test"],
            optimizer_f=ae_optimizer_f,
            scheduler_f=ae_scheduler_f,
        )
        trained_model = trainer.train_v2(
            checkpoint_filename="vae",
            args=args,
            train_2ndloader=train_2ndloader,
            val_loader=val_loader,
        )
    elif args.task == "val":
        checkpoint = torch.load(args.model_ckpt_dir, weights_only=True)
        vae.load_state_dict(checkpoint)
        print("Model loaded successfully!")
        vae.to(args.device)

        _ = evaluate_model(
            model=vae,
            args=args,
            val_loader=val_loader,
        )
        mean_class1 = _["mean_class1"]
        mean_class2 = _["mean_class2"]
        metrics = _["metrics"]
        pprint(metrics)

        # save means
        with open(
            os.path.join(args.model_save_dir, "means_val.pkl"),
            "wb",
        ) as f:
            pickle.dump({"mean_class1": mean_class1, "mean_class2": mean_class2}, f)
        print("Class mean saved at:", args.model_save_dir)


if __name__ == "__main__":
    main()
