import json
import os
import torch
import torch.optim as optim
import shutil
import time
from tqdm import tqdm
import lib.utils as utils
import visdom
import numpy as np
import torch.optim as optim
from functools import partial
from datetime import datetime
from utils.eval import score_dataset
from utils.schedulers.delayed_sched import *
from utils.schedulers.cosine_annealing_with_warmup import *
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import csv
import pickle


def init_model_params(args, dataset):
    return {
        "pose_shape": (
            dataset["test"][0][0].shape
            if args.model_confidence
            else dataset["test"][0][0][:2].shape
        ),
        "hidden_dim": args.model_latent_dim,
        "actnorm_scale": 1.0,
        "flow_coupling": "affine",
        "LU_decomposed": True,
        "learn_top": False,
        "device": args.device,
        "model_dist": "normal",
    }


def dump_args(args, ckpt_dir):
    path = os.path.join(ckpt_dir, "args.json")
    data = vars(args)
    with open(path, "w") as fp:
        json.dump(data, fp)


def calc_reg_loss(model, reg_type="l2", avg=True):
    reg_loss = None
    parameters = list(
        param for name, param in model.named_parameters() if "bias" not in name
    )
    num_params = len(parameters)
    if reg_type.lower() == "l2":
        for param in parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param**2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2) ** 2

        if avg:
            reg_loss /= num_params
        return reg_loss
    else:
        return torch.tensor(0.0, device=model.device)


def get_fn_suffix(args):
    fn_suffix = args.dataset + args.conv_oper
    return fn_suffix


class Trainer:
    def __init__(
        self,
        args,
        model,
        train_loader,
        test_loader,
        optimizer_f=None,
        scheduler_f=None,
        fn_suffix="",
    ):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fn_suffix = fn_suffix  # For checkpoint filename
        # Loss, Optimizer and Scheduler

        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == "adam":
            if self.args.lr:
                return optim.Adam(
                    self.model.parameters(),
                    lr=self.args.lr,
                    weight_decay=self.args.weight_decay,
                )
            else:
                return optim.Adam(self.model.parameters())
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
            )

    def adjust_lr(self, epoch, lr=None):
        if self.scheduler is not None:
            self.scheduler.step()
            new_lr = self.scheduler.get_lr()[0]
        elif (lr is not None) and (self.args.lr_decay is not None):
            new_lr = lr * (self.args.lr_decay**epoch)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
        else:
            raise ValueError("Missing parameters for LR adjustment")
        return new_lr

    def save_checkpoint(self, epoch, args, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = "checkpoint.pth.tar"

        state["args"] = args
        if not os.path.exists(self.args.save_dir):
            # Create the directory
            os.makedirs(self.args.save_dir)

        current_time = datetime.now()
        # path_join = os.path.join(self.args.ckpt_dir, filename)
        # path_join = os.path.join(self.args.save_dir, filename + '_' +  current_time.strftime("%Y-%m-%d_%H-%M-%S")+".pth.tar")
        path_join = os.path.join(
            self.args.save_dir, filename + "_" + str(epoch) + ".pth.tar"
        )
        torch.save(state, path_join)
        if is_best:
            # shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))
            shutil.copy(
                path_join, os.path.join(self.args.save_dir, "checkpoint_best.pth.tar")
            )

    def load_checkpoint(self, filename):
        filename = self.args.ckpt_dir + filename
        try:
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "Checkpoint loaded successfully from '{}' at (epoch {})\n".format(
                    self.args.ckpt_dir, checkpoint["epoch"]
                )
            )
        except FileNotFoundError:
            print(
                "No checkpoint exists from '{}'. Skipping...\n".format(
                    self.args.ckpt_dir
                )
            )

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if hasattr(self.model, "num_class"):
            checkpoint_state["n_classes"] = self.model.num_class
        if hasattr(self.model, "h_dim"):
            checkpoint_state["h_dim"] = self.model.h_dim
        return checkpoint_state

    def anneal_kl(self, iteration):

        warmup_iter = 300

        if self.args.lambda_anneal:
            self.model.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
        else:
            self.model.lamb = 0
        if self.args.beta_anneal:
            self.model.beta = min(
                self.args.beta, self.args.beta / warmup_iter * iteration
            )  # 0 --> 1
        else:
            self.model.beta = self.args.beta

    def plot_elbo(train_elbo, vis):
        global win_train_elbo
        win_train_elbo = vis.line(
            torch.Tensor(train_elbo), opts={"markers": True}, win=win_train_elbo
        )

    def train(self, num_epochs=None, log=True, checkpoint_filename=None, args=None):
        best_loss = -1e9
        train_elbo = []
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + self.fn_suffix + "_checkpoint.pth.tar"
        if (
            num_epochs is None
        ):  # For manually setting number of epochs, i.e. for fine tuning
            start_epoch = self.args.start_epoch
            num_epochs = args.epochs
        else:
            start_epoch = 0

        self.model = self.model.to(args.device)
        dataset_size = len(self.train_loader.dataset)
        elbo_running_mean = utils.RunningAverageMeter()
        if args.visdom:
            vis = visdom.Visdom(env=args.save, port=4500)
        it = 0
        for epoch in range(start_epoch, num_epochs):
            print("Started epoch {}".format(epoch))
            self.model.train()
            loss = []
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                it = it + 1
                data = data_arr[0].to(args.device, non_blocking=True)
                data = data[:, 0:2, :, :]
                self.anneal_kl(it)
                self.optimizer.zero_grad()
                mtime_1 = time.time()
                obj, elbo = self.model.elbo(data, dataset_size)
                mtime_2 = time.time()
                pbar.set_postfix(
                    elbo_time=mtime_2 - mtime_1, elbo_fps=1 / (mtime_2 - mtime_1)
                )

                if utils.isnan(obj).any():
                    raise ValueError("NaN spotted in objective.")
                obj.mean().mul(-1).backward()
                elbo_running_mean.update(elbo.mean())
                self.optimizer.step()
                loss.append(elbo_running_mean.avg)

            print(
                "[Epoch %03d] \tbeta %.2f \tlambda %.2f training ELBO: %.4f "
                % (epoch, self.model.beta, self.model.lamb, torch.stack(loss).mean())
            )
            new_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = self.adjust_lr(epoch, new_lr)
            print("lr: {0:.3e}".format(new_lr))
            train_elbo.append(torch.stack(loss).mean)

            if torch.stack(loss).mean() > best_loss:
                best_loss = torch.stack(loss).mean()
            self.save_checkpoint(epoch, args=args, filename=checkpoint_filename)
            print("Model saved!")
            eval_loss = []
            dataset_size = len(self.test_loader.dataset)
            self.model.eval()
            with torch.no_grad():
                for i, data_batch in enumerate(tqdm(self.test_loader)):
                    data = data_batch[0].to(args.device, non_blocking=True)
                    data = data[:, 0:2, :, :]
                    obj, elbo = self.model.elbo(data, dataset_size)
                    eval_loss.extend(elbo.cpu().numpy())
            auc_roc, dp_shift, dp_sigma, auc_pr, eer, eer_th = score_dataset(
                args.mask_root,
                np.array(eval_loss),
                self.test_loader.dataset.metadata,
                save_results=False,
                seg_len=args.seg_len,
            )
            print("AUC ROC: {}".format(auc_roc))
            print("AUC PR: {}".format(auc_pr))
            print("EER: {}".format(eer))
            print("EER TH: {}".format(eer_th))

        if args.visdom:
            self.plot_elbo(train_elbo, vis)

        return checkpoint_filename

    def train_v2(
        self,
        num_epochs=None,
        log=True,
        checkpoint_filename=None,
        args=None,
        train_2ndloader=None,
        val_loader=None,
    ):
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + self.fn_suffix + "_checkpoint.pth.tar"
        if (
            num_epochs is None
        ):  # For manually setting number of epochs, i.e. for fine tuning
            start_epoch = self.args.start_epoch
            num_epochs = args.epochs
        else:
            start_epoch = 0

        self.model = self.model.to(args.device)
        it = 0

        train_2nditer = iter(train_2ndloader)

        metrics = {
            "by_l2": {
                "pre": {"val": [], "best": 0.0},
                "rec": {"val": [], "best": 0.0},
                "f1": {"val": [], "best": 0.0},
                "acc": {"val": [], "best": 0.0},
                "cf_matrix": {"val": [], "best": None},
            },
            "by_logdensity": {
                "pre": {"val": [], "best": 0.0},
                "rec": {"val": [], "best": 0.0},
                "f1": {"val": [], "best": 0.0},
                "acc": {"val": [], "best": 0.0},
                "cf_matrix": {"val": [], "best": None},
            },
            "by_mean": {
                "pre": {"val": [], "best": 0.0},
                "rec": {"val": [], "best": 0.0},
                "f1": {"val": [], "best": 0.0},
                "acc": {"val": [], "best": 0.0},
                "cf_matrix": {"val": [], "best": None},
            },
        }

        results = []
        result_file = open(os.path.join(self.args.save_dir, "results.csv"), "w")
        csv_writer = csv.writer(result_file)

        for epoch in range(start_epoch, num_epochs):
            print("Started epoch {}".format(epoch))
            self.model.train()
            ls_running_loss = []
            ls_running_loss_rec = []
            ls_running_loss_kl = []
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                it = it + 1
                # this time, train_loader has only 1 class (normal)
                data_class1 = data_arr[0].to(args.device, non_blocking=True)
                data_class1 = data_class1[:, 0:2, :, :]
                data_class1 = data_class1.to(torch.float32)

                # train_2ndloader has both classes (normal and anomaly)
                try:
                    data_2nd, labels_2nd = next(train_2nditer)
                except StopIteration:
                    train_2nditer = iter(train_2ndloader)
                    data_2nd, labels_2nd = next(train_2nditer)
                data_2nd = data_2nd.to(args.device, non_blocking=True)
                labels_2nd = labels_2nd.to(args.device, non_blocking=True)

                labels_1st = torch.zeros(data_class1.shape[0], device=args.device)

                data = torch.cat((data_class1, data_2nd), dim=0)
                labels = torch.cat((labels_1st, labels_2nd), dim=0)
                # data = data_2nd
                # labels = labels_2nd

                data_class1 = data[labels == 0]
                data_class2 = data[labels == 1]

                self.optimizer.zero_grad()
                mtime_1 = time.time()
                _ = self.model.elbo_v2(
                    x_class1=data_class1,
                    target_mu_class1=-5,
                    target_logvar_class1=0,
                    x_class2=data_class2,
                    target_mu_class2=5,
                    target_logvar_class2=0,
                )
                loss = _["loss"]
                loss_rec = _["loss_rec"]
                loss_kl = _["loss_kl"]

                mtime_2 = time.time()
                pbar.set_postfix(
                    elbo_fps=1 / (mtime_2 - mtime_1),
                    loss_rec=loss_rec.item(),
                    loss_kl=loss_kl.item(),
                    loss=loss.item(),
                )

                if utils.isnan(loss).any():
                    raise ValueError("NaN spotted in objective.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
                self.optimizer.step()

                ls_running_loss.append(loss.item())
                ls_running_loss_rec.append(loss_rec.item())
                ls_running_loss_kl.append(loss_kl.item())

                # if itern == 20:
                #     break

            print(
                "[Epoch %03d] loss: %.3f, rec_loss: %.3f, kl_loss: %.3f "
                % (
                    epoch,
                    np.mean(ls_running_loss),
                    np.mean(ls_running_loss_rec),
                    np.mean(ls_running_loss_kl),
                )
            )
            new_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = self.adjust_lr(epoch, new_lr)
            print("lr: {0:.3e}".format(new_lr))

            # validation
            _ = evaluate_model(self.model, args, val_loader)
            mean_class1 = _["mean_class1"]
            mean_class2 = _["mean_class2"]
            metrics["by_l2"]["pre"]["val"].append(_["metrics"]["by_l2"]["pre"]["val"])
            metrics["by_l2"]["rec"]["val"].append(_["metrics"]["by_l2"]["rec"]["val"])
            metrics["by_l2"]["f1"]["val"].append(_["metrics"]["by_l2"]["f1"]["val"])
            metrics["by_l2"]["acc"]["val"].append(_["metrics"]["by_l2"]["acc"]["val"])
            metrics["by_l2"]["cf_matrix"]["val"].append(
                _["metrics"]["by_l2"]["cf_matrix"]["val"]
            )
            metrics["by_logdensity"]["pre"]["val"].append(
                _["metrics"]["by_logdensity"]["pre"]["val"]
            )
            metrics["by_logdensity"]["rec"]["val"].append(
                _["metrics"]["by_logdensity"]["rec"]["val"]
            )
            metrics["by_logdensity"]["f1"]["val"].append(
                _["metrics"]["by_logdensity"]["f1"]["val"]
            )
            metrics["by_logdensity"]["acc"]["val"].append(
                _["metrics"]["by_logdensity"]["acc"]["val"]
            )
            metrics["by_logdensity"]["cf_matrix"]["val"].append(
                _["metrics"]["by_logdensity"]["cf_matrix"]["val"]
            )
            metrics["by_mean"]["pre"]["val"].append(
                _["metrics"]["by_mean"]["pre"]["val"]
            )
            metrics["by_mean"]["rec"]["val"].append(
                _["metrics"]["by_mean"]["rec"]["val"]
            )
            metrics["by_mean"]["f1"]["val"].append(_["metrics"]["by_mean"]["f1"]["val"])
            metrics["by_mean"]["acc"]["val"].append(
                _["metrics"]["by_mean"]["acc"]["val"]
            )
            metrics["by_mean"]["cf_matrix"]["val"].append(
                _["metrics"]["by_mean"]["cf_matrix"]["val"]
            )
            results.append(
                {
                    "epoch": epoch,
                    "loss_rec": np.mean(ls_running_loss_rec),
                    "loss_kl": np.mean(ls_running_loss_kl),
                    "loss": np.mean(ls_running_loss),
                    "lr": new_lr,
                    "l2_acc": metrics["by_l2"]["acc"]["val"][-1],
                    "l2_f1": metrics["by_l2"]["f1"]["val"][-1],
                    "l2_pre": metrics["by_l2"]["pre"]["val"][-1],
                    "l2_rec": metrics["by_l2"]["rec"]["val"][-1],
                    "logdensity_acc": metrics["by_logdensity"]["acc"]["val"][-1],
                    "logdensity_f1": metrics["by_logdensity"]["f1"]["val"][-1],
                    "logdensity_pre": metrics["by_logdensity"]["pre"]["val"][-1],
                    "logdensity_rec": metrics["by_logdensity"]["rec"]["val"][-1],
                    "mean_acc": metrics["by_mean"]["acc"]["val"][-1],
                    "mean_f1": metrics["by_mean"]["f1"]["val"][-1],
                    "mean_pre": metrics["by_mean"]["pre"]["val"][-1],
                    "mean_rec": metrics["by_mean"]["rec"]["val"][-1],
                }
            )

            # save best model
            for cri in metrics:
                for met in metrics[cri]:
                    if (
                        met != "cf_matrix"
                        and metrics[cri][met]["val"][-1] > metrics[cri][met]["best"]
                    ):
                        metrics[cri][met]["best"] = metrics[cri][met]["val"][-1]
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(
                                self.args.save_dir, f"tsgad--best-{cri}-{met}.pth"
                            ),
                        )
                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=metrics[cri]["cf_matrix"]["val"][-1],
                            display_labels=["normal", "shoplifting"],
                        )
                        disp.plot(cmap=plt.cm.Blues)
                        plt.title(
                            f"Confusion matrix - best {cri} {met} (epoch {epoch})"
                        )
                        plt.savefig(
                            os.path.join(
                                self.args.save_dir,
                                f"confusion_matrix--best-{cri}-{met}.png",
                            )
                        )

            # write results to csv
            if len(results) == 1:
                csv_writer.writerow(sorted(list(results[-1].keys())))
            csv_writer.writerow([results[-1][k] for k in sorted(results[-1].keys())])
            result_file.flush()

            # save last model
            torch.save(
                self.model.state_dict(),
                os.path.join(self.args.save_dir, f"tsgad--last.pth"),
            )

            for cri in metrics:
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=metrics[cri]["cf_matrix"]["val"][-1],
                    display_labels=["normal", "shoplifting"],
                )
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f"Confusion matrix - last (epoch {epoch})")
                plt.savefig(
                    os.path.join(
                        self.args.save_dir, f"confusion_matrix--last-{cri}.png"
                    )
                )

            # save means
            with open(
                os.path.join(self.args.save_dir, "means_epoch_{}.pkl".format(epoch)),
                "wb",
            ) as f:
                pickle.dump({"mean_class1": mean_class1, "mean_class2": mean_class2}, f)

        result_file.close()

        return checkpoint_filename


def evaluate_model(model, args, val_loader):
    model.eval()

    mean_class1 = []
    mean_class2 = []

    metrics = {}

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(val_loader)):
            data = data.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            data_class1 = data[labels == 0]
            data_class2 = data[labels == 1]

            _, z_params_class1, _ = model.encode_v2(data_class1)
            if len(data_class1) > 0:
                mean_class1.extend(z_params_class1[:, :, 0].cpu().numpy())

            _, z_params_class2, _ = model.encode_v2(data_class2)
            if len(data_class2) > 0:
                mean_class2.extend(z_params_class2[:, :, 0].cpu().numpy())
    mean_class1 = np.mean(mean_class1, axis=0)
    mean_class2 = np.mean(mean_class2, axis=0)

    eval_l2 = []
    eval_logdensity = []
    eval_mean = []
    eval_lbl = []
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(val_loader)):
            data = data.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            data_class1 = data[labels == 0]
            data_class2 = data[labels == 1]

            _, z_params_class1, _ = model.encode_v2(data_class1)
            _, z_params_class2, _ = model.encode_v2(data_class2)

            eval_l2_class1 = np.argmin(
                np.stack(
                    [
                        np.sum(
                            (z_params_class1[:, :, 0].cpu().numpy() - mean_class1) ** 2,
                            axis=1,
                        ),
                        np.sum(
                            (z_params_class1[:, :, 0].cpu().numpy() - mean_class2) ** 2,
                            axis=1,
                        ),
                    ],
                    axis=1,
                ),
                axis=1,
            )
            eval_l2_class2 = np.argmin(
                np.stack(
                    [
                        np.sum(
                            (z_params_class2[:, :, 0].cpu().numpy() - mean_class1) ** 2,
                            axis=1,
                        ),
                        np.sum(
                            (z_params_class2[:, :, 0].cpu().numpy() - mean_class2) ** 2,
                            axis=1,
                        ),
                    ],
                    axis=1,
                ),
                axis=1,
            )

            eval_logdensity_class1 = np.argmax(
                np.stack(
                    [
                        log_density(
                            input_=z_params_class1[:, :, 0].cpu().numpy(),
                            target_mu=-5,
                            target_logvar=0,
                        ),
                        log_density(
                            input_=z_params_class1[:, :, 0].cpu().numpy(),
                            target_mu=5,
                            target_logvar=0,
                        ),
                    ],
                    axis=1,
                ),
                axis=1,
            )
            eval_logdensity_class2 = np.argmax(
                np.stack(
                    [
                        log_density(
                            input_=z_params_class2[:, :, 0].cpu().numpy(),
                            target_mu=-5,
                            target_logvar=0,
                        ),
                        log_density(
                            input_=z_params_class2[:, :, 0].cpu().numpy(),
                            target_mu=5,
                            target_logvar=0,
                        ),
                    ],
                    axis=1,
                ),
                axis=1,
            )

            eval_mean_class1 = np.argmin(
                np.stack(
                    [
                        np.mean(
                            np.abs(z_params_class1[:, :, 0].cpu().numpy() - (-5)),
                            axis=1,
                        ),
                        np.mean(
                            np.abs(z_params_class1[:, :, 0].cpu().numpy() - 5),
                            axis=1,
                        ),
                    ],
                    axis=1,
                ),
                axis=1,
            )
            eval_mean_class2 = np.argmax(
                np.stack(
                    [
                        np.mean(
                            np.abs(z_params_class2[:, :, 0].cpu().numpy() - (-5)),
                            axis=1,
                        ),
                        np.mean(
                            np.abs(z_params_class2[:, :, 0].cpu().numpy() - 5),
                            axis=1,
                        ),
                    ],
                    axis=1,
                ),
                axis=1,
            )

            eval_l2.extend(eval_l2_class1.tolist() + eval_l2_class2.tolist())
            eval_logdensity.extend(
                eval_logdensity_class1.tolist() + eval_logdensity_class2.tolist()
            )
            eval_mean.extend(eval_mean_class1.tolist() + eval_mean_class2.tolist())
            eval_lbl.extend([0] * len(data_class1) + [1] * len(data_class2))

    eval_lbl = np.array(eval_lbl)
    eval_l2 = np.array(eval_l2)
    eval_logdensity = np.array(eval_logdensity)
    eval_mean = np.array(eval_mean)

    _ = eval_metrics(eval_lbl, eval_l2)
    metrics["by_l2"] = {
        "pre": {"val": _["precision"]},
        "rec": {"val": _["recall"]},
        "f1": {"val": _["f1"]},
        "acc": {"val": _["accuracy"]},
        "cf_matrix": {"val": _["confusion_matrix"]},
    }

    _ = eval_metrics(eval_lbl, eval_logdensity)
    metrics["by_logdensity"] = {
        "pre": {"val": _["precision"]},
        "rec": {"val": _["recall"]},
        "f1": {"val": _["f1"]},
        "acc": {"val": _["accuracy"]},
        "cf_matrix": {"val": _["confusion_matrix"]},
    }

    _ = eval_metrics(eval_lbl, eval_mean)
    metrics["by_mean"] = {
        "pre": {"val": _["precision"]},
        "rec": {"val": _["recall"]},
        "f1": {"val": _["f1"]},
        "acc": {"val": _["accuracy"]},
        "cf_matrix": {"val": _["confusion_matrix"]},
    }

    return {
        "mean_class1": mean_class1,
        "mean_class2": mean_class2,
        "metrics": metrics,
    }


def init_optimizer(type_str, **kwargs):
    if type_str.lower() == "adam":
        opt_f = optim.Adam
    else:
        return None

    return partial(opt_f, **kwargs)


def init_scheduler(type_str, lr, epochs, warmup=3):
    sched_f = None
    if type_str.lower() == "exp_decay":
        sched_f = None
    elif type_str.lower() == "cosine":
        sched_f = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=epochs)
    elif type_str.lower() == "cosine_warmup":
        sched_f = partial(CosineAnnealingWarmUpRestarts, T_0=epochs, T_up=warmup)
    elif type_str.lower() == "cosine_delayed":
        sched_f = partial(
            DelayedCosineAnnealingLR,
            delay_epochs=warmup,
            cosine_annealing_epochs=epochs,
        )
    elif (type_str.lower() == "tri") and (epochs >= 8):
        sched_f = partial(
            optim.lr_scheduler.CyclicLR,
            base_lr=lr,
            max_lr=lr,
            step_size_up=epochs // 8,
            mode="triangular2",
            cycle_momentum=False,
        )
    else:
        print("Unable to initialize scheduler, defaulting to exp_decay")

    return sched_f


def eval_metrics(eval_lbl, eval_pred):
    precision = precision_score(eval_lbl, eval_pred, average="binary", pos_label=1)
    recall = recall_score(eval_lbl, eval_pred, average="binary", pos_label=1)
    f1 = f1_score(eval_lbl, eval_pred, average="binary", pos_label=1)
    acc = accuracy_score(eval_lbl, eval_pred)
    cf_matrix = confusion_matrix(eval_lbl, eval_pred, labels=[0, 1], normalize=None)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "confusion_matrix": cf_matrix,
    }


def log_density(**kwargs):
    input_ = kwargs["input_"]  # shape (N, M)
    target_mu = kwargs["target_mu"]
    target_logvar = kwargs["target_logvar"]

    return np.sum(
        -0.5 * np.log(2 * np.pi)
        - 0.5 * target_logvar
        - (input_ - target_mu) ** 2 / (2 * np.exp(target_logvar)),
        axis=1,
    )
