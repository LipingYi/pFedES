import argparse
import copy
import csv
import json
import logging
import math
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
import sys


import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from tqdm import trange

from experiments.FLHetero.Models.CNNs import CNN_1, CNN_2, CNN_3, CNN_4, CNN_5, Generator



from experiments.FLHetero.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool
import pickle



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




random.seed(2022)

def test_acc(net, testloader,criteria):
    net.eval()
    with torch.no_grad():
        test_acc = 0
        num_batch = 0

        for batch in testloader:
            num_batch += 1
            # batch = next(iter(testloader))
            img, label = tuple(t.to(device) for t in batch)
            pred, _ = net(img)
            test_loss = criteria(pred, label)
            test_acc += pred.argmax(1).eq(label).sum().item() / len(label)
        mean_test_loss = test_loss / num_batch
        mean_test_acc = test_acc / num_batch
    return mean_test_loss, mean_test_acc



def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int, fraction: float,
          steps: int, epochs: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int, total_classes: int) -> None:

    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)

    # -------compute aggregation weights-------------#
    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    client_sample_count = [train_sample_count[i] + eval_sample_count[i] + test_sample_count[i] for i in
                           range(len(train_sample_count))]
    # -----------------------------------------------#


    print(data_name)
    if data_name == "cifar10":
        net_1 = CNN_1(n_kernels=n_kernels)
        net_2 = CNN_2(n_kernels=n_kernels)
        net_3 = CNN_3(n_kernels=n_kernels)
        net_4 = CNN_4(n_kernels=n_kernels)
        net_5 = CNN_5(n_kernels=n_kernels)
        generator = Generator()
    elif data_name == "cifar100":
        net_1 = CNN_1(n_kernels=n_kernels, out_dim=100)
        net_2 = CNN_2(n_kernels=n_kernels, out_dim=100)
        net_3 = CNN_3(n_kernels=n_kernels, out_dim=100)
        net_4 = CNN_4(n_kernels=n_kernels, out_dim=100)
        net_5 = CNN_5(n_kernels=n_kernels, out_dim=100)
        generator = Generator()
    elif data_name == "mnist":
        net = CNN_1(n_kernels=n_kernels)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

    net_1 = net_1.to(device)
    net_2 = net_2.to(device)
    net_3 = net_3.to(device)
    net_4 = net_4.to(device)
    net_5 = net_5.to(device)

    net_set = [net_1, net_2, net_3, net_4, net_5]
    generator = generator.to(device)

    init_gen_paras = copy.deepcopy(generator.state_dict())


    ##################
    # init optimizer #
    ##################

    # optimizers = {
    #     'sgd': torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=wd),
    #     'adam': torch.optim.Adam(params=net.parameters(), lr=lr)
    # }
    # optimizer = optimizers[optim]
    # optimizer_gh = optimizer

    # lr_gh = 3 # [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    # optimizer_gh = torch.optim.SGD(params=net.parameters(), lr=lr_gh, momentum=0.9, weight_decay=wd)

    lr_net = 0.01 #[optimal]  # [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    lr_gen = 0.01 # 0.001, 0.01, 0.1
    optimizer_gen = torch.optim.SGD(params=generator.parameters(), lr=lr_gen, momentum=0.9, weight_decay=wd)

    criteria = torch.nn.CrossEntropyLoss()

    epoch_net = epochs
    epoch_generator = epochs+10  # -5, +5, +10


    ################
    # init metrics #
    ################
    step_iter = trange(steps)


    PM_acc = defaultdict()
    PMs = defaultdict()
    Data_Distirbutions = defaultdict()
    LGs = defaultdict() # local generators
    GG = init_gen_paras # global generator
    MIU = defaultdict()


    for i in range(num_nodes):
        PM_acc[i] = 0
        PMs[i] = copy.deepcopy(net_set[i%5].state_dict())
        LGs[i] = init_gen_paras
        Data_Distirbutions[i] = defaultdict(int)
        MIU[i] = 0.1


    miu = 0.5

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / f"pFedES_mh_miu_{miu}_GEpo_{epoch_generator}_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}_class_{classes_per_node}.csv"), 'w', newline='') as file:

        mywriter = csv.writer(file, delimiter=',')


        for step in step_iter:  # step is round
            round_id = step
            frac = fraction
            select_nodes = random.sample(range(num_nodes), int(frac * num_nodes))

            all_local_trained_loss = []
            all_local_trained_acc = []
            results = []


            logging.info(f'#----Round:{step}----#')
            for c in select_nodes:
                node_id = c
                print(f'client id: {node_id}')

                net = net_set[node_id % 5]
                optimizer_net = torch.optim.SGD(params=net.parameters(), lr=lr_net, momentum=0.9, weight_decay=wd)

                net.load_state_dict(PMs[node_id])
                generator.load_state_dict(GG)



                # local training
                # 1. freeze generator, train net
                net.train()
                for i in range(epoch_net):
                    for j, batch in enumerate(nodes.train_loaders[node_id], 0):
                        # print(f'batch is {j}')
                        img, label = tuple(t.to(device) for t in batch)

                        optimizer_net.zero_grad()

                        _, img_gen = generator(img)

                        pred, rep = net(img)
                        pred_gen, _ = net(img_gen)

                        hard_loss = criteria(pred, label)
                        # contrstive_loss = F.kl_div(torch.log(pred.softmax(dim=-1) + 1e-5),
                        #                            pred_gen.detach().softmax(dim=-1),
                        #                            reduction='sum')
                        # miu = max(np.exp(-step/ 100), 0.5)
                        # # y = [max(np.exp(-i / 100), 0.5) for i in x]
                        # loss = (1-miu)*contrstive_loss + miu*hard_loss

                        # miu = min(round_id/steps, 0.5)

                        hard_loss_gen = criteria(pred_gen, label)
                        loss = hard_loss * (1-miu) + hard_loss_gen * miu
                        # loss = hard_loss + hard_loss_gen

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
                        optimizer_net.step()
                # print(f'miu:{miu}')
                PMs[node_id] = copy.deepcopy(net.state_dict())

                # 2. freeze net, train generator
                generator.train()
                for i in range(epoch_generator):
                    for j, batch in enumerate(nodes.train_loaders[node_id], 0):
                        # print(f'batch is {j}')
                        img, label = tuple(t.to(device) for t in batch)

                        optimizer_gen.zero_grad()

                        _, img_gen = generator(img)

                        # pred, rep = net(img)
                        pred_gen, _ = net(img_gen)

                        loss = criteria(pred_gen, label)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), 50)
                        optimizer_gen.step()

                LGs[node_id] = copy.deepcopy(generator.state_dict())


                # evaluate trained local model
                trained_loss, trained_acc = test_acc(net, nodes.test_loaders[node_id], criteria)
                all_local_trained_loss.append(trained_loss.cpu().item())
                all_local_trained_acc.append(trained_acc)
                PM_acc[node_id] = trained_acc

                logging.info(f'Round {step} | client {node_id} acc: {PM_acc[node_id]}')


            mean_trained_loss = round(np.mean(all_local_trained_loss), 4)
            mean_trained_acc = round(np.mean(all_local_trained_acc), 4)
            results.append([mean_trained_loss, mean_trained_acc] + [round(i,4) for i in PM_acc.values()])
            mywriter.writerows(results)
            file.flush()

            logging.info(f'Round:{step} | mean_trained_loss:{mean_trained_loss} | mean_trained_acc:{mean_trained_acc}')


            # aggregate local generators to update global generator
            client_agg_weights = OrderedDict()
            select_nodes_sample_count = OrderedDict()
            for i in range(len(select_nodes)):
                select_nodes_sample_count[select_nodes[i]] = client_sample_count[select_nodes[i]]
            for i in range(len(select_nodes)):
                client_agg_weights[select_nodes[i]] = select_nodes_sample_count[select_nodes[i]] / sum(
                    select_nodes_sample_count.values())

            weight_keys = list(generator.state_dict().keys())
            # generator_paras = OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for id, model in LGs.items():
                    if id in select_nodes:
                        key_sum += client_agg_weights[id] * model[key]
                GG[key] = key_sum
            logging.info(f'Global generator is updated after aggregation')


        logging.info('Federated Learning has been successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Learning with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar100", choices=['cifar10', 'cifar100', 'mnist'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--total-classes", type=str, default=100)
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")
    parser.add_argument("--fraction", type=int, default=0.1, help="number of sampled nodes in each round")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-3, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="Results/FedDGS", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2 # 2, 4, 6, 8, 10
    elif args.data_name == 'cifar100':
        args.classes_per_node = 10 # 30, 50, 70, 90, 100
    else:
        args.classes_per_node = 2

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        total_classes = args.total_classes,
        num_nodes=args.num_nodes,
        fraction=args.fraction,
        steps=args.num_steps,
        epochs=args.epochs,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed
    )
