from args import create_exp_dirs
from args import init_parser, init_sub_args
import torch
import random
import numpy as np
import lib.dist as dist
from lib.flows import FactorialNormalizingFlow
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params, Trainer, init_optimizer, init_scheduler
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
    features = (features[..., :2] - features[..., :2].mean(axis=(1, 2))[:, None, None, :]) / features[..., 1].std(axis=(1, 2))[:, None, None, None]
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

TRAIN_FILES_1 = (
    [
        '/home/laptq/laptq-fs26-shoplifting-detection/outputs/create_input_train_dataset_for_TSGAD/poselift/'
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
)


def main ():
    print('sedaye mano darid az Chalotte America!')
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
    
    pretrained_model = vars(args).get('model_ckpt_dir', None)
    
    if not os.path.exists('/home/laptq/laptq-fs26-shoplifting-detection/outputs/trivials/tsgad-data.pkl'):
        dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained_model is not None))
        with open('/home/laptq/laptq-fs26-shoplifting-detection/outputs/trivials/tsgad-data.pkl', 'wb') as f:
            pickle.dump((dataset, loader), f)
    else:
        with open('/home/laptq/laptq-fs26-shoplifting-detection/outputs/trivials/tsgad-data.pkl', 'rb') as f:
            dataset, loader = pickle.load(f)
    # dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained_model is not None))
    
    # model_args = init_model_params(args, dataset)

    # laptq
    train_2ndloader, _ = load_dataset(
        # data_files[: int(train_ratio * len(data_files))],
        TRAIN_FILES_1,
        64,
        shuffle=True,
        split_size=0,
        sample_balance=False,
    )
    train_2nditer = iter(train_2ndloader)
    
    prior_dist = dist.Normal(mu=0)
    q_dist = dist.Normal(mu=0)

    vae = VAE(z_dim=model_args.latent_dim, 
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
    
    if pretrained_model == None:
        if not os.path.exists(args.model_save_dir):
        # Create the directory
            os.makedirs(args.model_save_dir)
        arguments = vars(args)
        with open(args.model_save_dir + '/' + 'arguments.yaml', 'w') as file:
            yaml.dump(arguments, file)
        ae_optimizer_f = init_optimizer(args.model_optimizer, lr=args.model_lr)
        ae_scheduler_f = init_scheduler(args.sched, lr=args.model_lr, epochs=args.epochs)
        trainer = Trainer(model_args, vae, loader['train'], loader['test'], optimizer_f=ae_optimizer_f,
                                scheduler_f=ae_scheduler_f)
        trained_model = trainer.train_v2(checkpoint_filename='vae', args=args, train_2ndloader=train_2ndloader)
        
    else:
        checkpoint = torch.load(args.model_ckpt_dir, weights_only=False)
        vae.load_state_dict(checkpoint['state_dict'])
        print('Model loaded successfully!')
        vae.to(args.device)
        
    eval_loss = []
    eval_elbo = []
    dataset_size = len(loader['test'].dataset)
    mean, std = get_train_dist (vae, loader['test'], args)
    model_id = os.path.split(args.model_ckpt_dir)[-1]
    # laptq
    # with open('/home/laptq/laptq-fs26-shoplifting-detection/outputs/TSGAD4/cached-scores/TSGAD-mean-test-{}.pkl'.format(model_id), 'wb') as f:
    #     pickle.dump(mean, f)
    mean = torch.from_numpy(mean).to(args.device)

    # distribution_m = norm(loc=m_mean, scale=m_std)
    # distribution_v = norm(loc=v_mean, scale=v_std)
    # distribution = multivariate_normal(mean=mean.astype(np.float64), cov=np.diag(std.astype(np.float64)**2))

    vae.eval()
    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(loader['test'])):
            data_1st = data_batch[0].to(args.device, non_blocking=True)
            data_1st = data_1st[:,0:2, :, :]
            obj, elbo = vae.elbo(data_1st, dataset_size)
            eval_elbo.extend(elbo.cpu().numpy())
            data_1st = data_1st.view(data_1st.shape[0], 2, args.seg_len, 17)
            _, z_params, _ = vae.encode(data_1st)
            z_params = z_params.view(z_params.shape[0], -1)
            l2_distance = (torch.sqrt(torch.sum((z_params - mean)**2, dim=1))).cpu().numpy()
            eval_loss.extend(l2_distance)
            # probability_m = distribution_m.pdf(z_params[:, :, 0])
            # probability_v = distribution_v.pdf(z_params[:, :, 1])
            # probability = distribution.pdf(z_params.view(data.shape[0], -1).cpu().astype(np.float64))
            
            # Calculate the joint probability by multiplying the probabilities of the two variables
            # joint_probability = probability_m * probability_v
    auc_roc, dp_shift, dp_sigma, auc_pr, eer, eer_th = score_dataset(args.mask_root, np.array(eval_loss), dataset['test'].metadata, save_results=args.save_scores, seg_len=args.seg_len, directory=args.score_save_dir, model_id=model_id)
    print("*** Normal Dist ***")
    print('AUC ROC: {}'.format(auc_roc))
    print('AUC PR: {}'.format(auc_pr))
    print('EER: {}'.format(eer))
    print('EER TH: {}'.format(eer_th))

    # auc_roc, dp_shift, dp_sigma, auc_pr, eer, eer_th = score_dataset(args.mask_root, np.array(eval_elbo), dataset['test'].metadata, save_results=False, seg_len=args.seg_len)
    # print("*** ELBO ***")
    # print('AUC ROC: {}'.format(auc_roc))
    # print('AUC PR: {}'.format(auc_pr))
    # print('EER: {}'.format(eer))
    # print('EER TH: {}'.format(eer_th))
    
if __name__ == '__main__':
    main()