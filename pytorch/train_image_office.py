import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import random
import pdb
import math

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    feature_out, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs) / 10
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    all_feature = feature_out.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_feature = torch.cat((all_feature, feature_out.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                feature_out, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    all_feature = feature_out.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_feature = torch.cat((all_feature, feature_out.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu() ), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, all_output.numpy(), predict.numpy(), all_label.numpy(), all_feature.numpy()


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    tensor_writer = SummaryWriter(config["tensorboard_path"])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1 or i==0:
            base_network.train(False)
            temp_acc, output, prediction, label, feature = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]                  
        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        outputs_target_temp = outputs_target / config['temperature']
        target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
        target_entropy_weight = loss.Entropy(target_softmax_out_temp).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(target_softmax_out_temp)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = classifier_loss + mcc_loss
        total_loss.backward()
        optimizer.step()

        tensor_writer.add_scalar('total_loss', total_loss, i)
        tensor_writer.add_scalar('classifier_loss', classifier_loss, i)
        tensor_writer.add_scalar('cov_matrix_penalty', mcc_loss, i)

    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='5', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home','DomainNet'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/dslr_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/amazon_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='office_temp2.5', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--temperature', type=float, default=2.5, help="temperature value in MCC")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # train config
    config = {}
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 50004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    task_name = args.output_dir + '/' + osp.basename(args.s_dset_path)[0].upper() + '2' + osp.basename(args.t_dset_path)[0].upper()
    config["output_path"] = "snapshot-eccv-opensource/" + task_name
    config["tensorboard_path"] = "vis/" + task_name
    config['temperature'] = args.temperature
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":28}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":28}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":4}}

    if config["dataset"] == "office":
        config["optimizer"]["lr_param"]["lr"] = 0.0003
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "DomainNet":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 345
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
