import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from torch.utils.data import DataLoader
import torch
import numpy as np
from pytorch_msssim import MS_SSIM, ssim
import random
from dataloader import get_dataset, get_datapath
from model.net import NestFuse_light2_nodense, Fusion_network
from tqdm import tqdm, trange
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from utils.gradient import gradient

epochs = 20
EPSILON = 1e-6
number = 0


# torch.autograd.set_detect_anomaly(True)
def _save_checkpoint(model, dir, name, save_best=False, overwrite=True):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()

    filename = os.path.join(dir, name + " best.pth")
    torch.save(checkpoint, filename)
    print("Saving current best model: best_model.pth")


def evaluate(
    nest_model,
    fusion_model,
    dataset_name,
    dataset_path,
    beta,
    gamma,
    w1,
    w2,
    ssim_vi,
    ssim_ir,
    tensorboard_writer,
    number=number,
    save_dir=None,
):
    if not hasattr(evaluate,"counter"):
        evaluate.counter=0
    dataset = get_dataset(dataset_name)(dataset_path)
    dataloader = DataLoader(dataset, batch_size=10, num_workers=4, drop_last=True)
    vi_ssim_loss = MS_SSIM(data_range=1.0, channel=1)
    ir_ssim_loss = MS_SSIM(data_range=1.0, channel=1)
    mse_loss = torch.nn.MSELoss()
    nest_model.eval()
    fusion_model.eval()
    ssim_list = []
    ms_ssim_list = []
    iter1 = tqdm(enumerate(dataloader), total=len(dataloader))
    for index, batch in iter1:
        img_vi, img_ir = batch
        img_vi = img_vi.cuda()[:, 0, :, :].unsqueeze(1)
        img_ir = img_ir.cuda()[:, 0, :, :].unsqueeze(1)
        loss1_value = 0
        loss2_value = 0
        with torch.no_grad():
            en_ir = nest_model.encoder(img_ir)
            en_vi = nest_model.encoder(img_vi)
            # fusion
            f = fusion_model(en_ir, en_vi)
            f = f + en_vi
            # decoder
            outputs = nest_model.decoder_eval(f)
            x_vi = Variable(img_vi.data.clone(), requires_grad=False)
            x_ir = Variable(img_ir.data.clone(), requires_grad=False)
            for output in outputs:
                ssim_loss_temp_vi = vi_ssim_loss(output, x_vi)
                ssim_loss_temp_ir = ir_ssim_loss(output, x_ir)
                ssim_vi = ssim(output, x_vi, data_range=1.0)
                ssim_ir = ssim(output, x_ir, data_range=1.0)
                loss1_value += ssim_vi * (1 - ssim_loss_temp_vi) + ssim_ir * (
                    1 - ssim_loss_temp_ir
                )

                # # feature loss
                g2_ir_fea = en_ir
                g2_vi_fea = en_vi
                g2_fuse_fea = f

                w_ir = [w1, w1, w1, w1]
                w_vi = [w2, w2, w2, w2]
                w_fea = [1, 10, 100, 1000]
                for ii in range(4):
                    g2_ir_temp = g2_ir_fea[ii]
                    g2_vi_temp = g2_vi_fea[ii]
                    g2_fuse_temp = g2_fuse_fea[ii]
                    (bt, cht, ht, wt) = g2_ir_temp.size()
                    loss2_value += w_fea[ii] * mse_loss(
                        g2_fuse_temp, w_ir[ii] * g2_ir_temp + w_vi[ii] * g2_vi_temp
                    )
                # loss2_value += mse_loss(output,w1*x_vi+w2*x_ir)
                gradinet_loss = F.l1_loss(
                    gradient(output), torch.max(gradient(x_ir), gradient(x_vi))
                )
            loss1_value /= len(outputs)
            loss2_value /= len(outputs)
            gradinet_loss /= len(output)
            iter1.set_postfix(
                count=number,
                loss_mse=0.07*loss2_value.item(),
                loss_texture=gamma * gradinet_loss.item(),
                loss_ssim=beta * loss1_value.item(),
            )
            total_loss = beta * loss1_value + loss2_value + gamma * gradinet_loss

            loss1_value = loss1_value.item()
            loss2_value = loss2_value.item()
            total_loss = loss1_value + loss2_value
            if index % 50 == 0:
                tensorboard_writer.add_scalar(
                    "Validation/total_loss", total_loss, evaluate.counter
                )
                tensorboard_writer.add_scalar(
                    "Validation/ms_ssim_loss",  loss1_value, evaluate.counter
                )
                tensorboard_writer.add_scalar(
                    "Validation/mse_loss", loss2_value, evaluate.counter
                )
                tensorboard_writer.add_scalar(
                    "Validation/texture_loss", gradinet_loss.item(), evaluate.counter
                )
                tensorboard_writer.add_scalar(
                    "Validation/vi_ms_ssim", ssim_loss_temp_vi.item(), evaluate.counter
                )
                tensorboard_writer.add_scalar(
                    "Validation/vi_ssim", ssim_vi.item(), evaluate.counter
                )
                tensorboard_writer.add_scalar(
                    "Validation/ir_ms_ssim", ssim_loss_temp_ir.item(), evaluate.counter
                )
                tensorboard_writer.add_scalar(
                    "Validation/ir_ssim", ssim_ir.item(), evaluate.counter
                )
                ssim_list.append(ssim_vi.item())
                ms_ssim_list.append(ssim_loss_temp_vi.item())
            evaluate.counter+=1

    return max(ssim_list), max(ms_ssim_list)


def save_img(img, path):
    batch = img.shape[0]
    for i in range(batch):
        img = img.detach()
        img_numpy = img[i].squeeze().cpu().numpy() * 255
        img_resize = cv2.resize(img_numpy, (640, 512), cv2.INTER_AREA)
        path1 = os.path.join(path, f"第{i}张图.png")
        cv2.imwrite(str(path1), img_resize)


def set_seed(seed=42):
    """为PyTorch、NumPy和Python内置的random库设置种子，以确保实验可重复性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    # 确保PyTorch的卷积操作的确定性，这可能会降低一些运行效率
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)  # 你可以选择任何喜欢的整数作为种子


def create_dir(beta, gamma, w1, w2,ssim_vi,ssim_ir):
    root_dir = r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved"
    path = Path(root_dir)
    dir_path = os.path.join(
        path,
        "beta:"
        + str(beta)
        + "--gamma:"
        + str(gamma)
        + "--w1:"
        + str(w1)
        + "--w2:"
        + str(w2)
        + "--ssim vi:"
        +str(ssim_vi)
        +"--ssim ir:"
        + str(ssim_ir)
    )
    dir_path = str(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def main():
    dataset = get_dataset("kaist")(get_datapath("kaist"))
    # True - RGB , False - gray
    img_flag = False
    beta_gamma_list = sorted([[5,70]])# [[15,700],[10,700],[15,900],[25,900]]
    weight_for_vi_if_list = [[6,3]]# [ [0.7, 0.3], [0.8, 0.2]]
    weight_for_vi_if_ssim_list=[[1.0,0.5],[0.5,0.5]]#,[0.5,0.5]]# [1.0,1.0],[0.5,1.0],[0.5,0.5]]

    for w_w in weight_for_vi_if_list:
        w1, w2 = w_w
        for ssim_vi,ssim_ir in weight_for_vi_if_ssim_list:
            for beta, gamma in beta_gamma_list:
                train(dataset, img_flag, beta, gamma, w1, w2,ssim_vi,ssim_ir)


def train(dataset, img_flag, beta, gamma, w1, w2,ssim_vi,ssim_ir):
    evaluate1 = evaluate
    evaluate1.counter = 0
    batch_size = 10
    # load network model
    nc = 1
    input_nc = nc
    output_nc = nc
    # nb_filter = [64, 128, 256, 512]
    nb_filter = [64, 112, 160, 208]
    f_type = "res"
    root_dir = create_dir(beta, gamma, w1, w2,ssim_vi,ssim_ir)
    pth_dir = os.path.join(
        "/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved",
        "beta:"
        + str(beta)
        + "--gamma:"
        + str(gamma)
        + "--w1:"
        + str(w1)
        + "--w2:"
        + str(w2)
        + "--ssim vi:"
        +str(ssim_vi)
        +"--ssim ir:"
        + str(ssim_ir)
    )
    nest = os.path.join(pth_dir,"nest model best.pth")
    fusion = os.path.join(pth_dir,"fusion model best.pth")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )

    deepsupervision = False
    nest_model = NestFuse_light2_nodense(
        nb_filter, input_nc, output_nc, deepsupervision
    )
    # model_path = r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/model/nestfuse_gray_1e2.model"
    # load auto-encoder network
    # print(
    #     "Resuming, initializing auto-encoder using weight from {}.".format(
    #         model_path
    #     )
    # )
    # nest = torch.load(nest)
    # nest_model.load_state_dict(nest["model"])
    nest_model.cuda()
    nest_model.eval()

    # fusion network
    fusion_model = Fusion_network(nb_filter, f_type)
    # fusion = torch.load(fusion)
    # fusion_model.load_state_dict(fusion["model"])
    fusion_model.cuda()
    fusion_model.train()
    nest_model.train()

    # if args.resume_fusion_model is not None:
    # 	print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_fusion_model))
    # 	fusion_model.load_state_dict(torch.load(args.resume_fusion_model))
    parameters = list(fusion_model.parameters()) + list(nest_model.parameters())
    optimizer = torch.optim.Adam(parameters, 7e-5)
    mse_loss = torch.nn.MSELoss()
    vi_ssim_loss = MS_SSIM(data_range=1.0, channel=1)
    ir_ssim_loss = MS_SSIM(data_range=1.0, channel=1)

    tbar = trange(epochs)
    print("Start training.....")
    # mode = args.mode
    # print(mode)
    # creating save path

    # temp_path_model = os.path.join(args.save_fusion_model)
    # temp_path_loss  = os.path.join(args.save_loss_dir)
    # if os.path.exists(temp_path_model) is False:
    # 	os.mkdir(temp_path_model)

    # if os.path.exists(temp_path_loss) is False:
    # 	os.mkdir(temp_path_loss)

    # temp_path_model_w = os.path.join(args.save_fusion_model, str(w1), mode)
    # temp_path_loss_w  = os.path.join(args.save_loss_dir, str(w1))
    # if os.path.exists(temp_path_model_w) is False:
    # 	os.mkdir(temp_path_model_w)

    # if os.path.exists(temp_path_loss_w) is False:
    # 	os.mkdir(temp_path_loss_w)

    # Loss_feature = []
    # Loss_ssim = []
    # Loss_all = []
    # count_loss = 0

    nest_model.cuda()
    # trans_model.cuda()
    fusion_model.cuda()
    # sobel_loss = nn.L1Loss()
    tensorboard_dir = f"{beta} {gamma} {w1} {w2} {ssim_vi} {ssim_ir}"
    tensorboard = "/root/tf-logs/" + tensorboard_dir
    tensor_writer = SummaryWriter(tensorboard, flush_secs=30)
    max_ssim = 0
    max_ms_ssim = 0
    count = 0
    for e in tbar:
        print("Epoch %d....." % e)
        # load training database
        iter1 = tqdm(enumerate(dataloader), total=len(dataloader))
        for index, batch in iter1:
            ssim = 0
            ms_ssim = 0
            all_ssim_loss = 0.0
            all_fea_loss = 0.0
            loss1_value = 0.0
            loss2_value = 0.0
            optimizer.zero_grad()
            img_vi, img_ir = batch
            img_ir = img_ir.cuda()[:, 0, :, :].unsqueeze(1)
            img_vi = img_vi.cuda()[:, 0, :, :].unsqueeze(1)
            # encoder
            en_ir = nest_model.encoder(img_ir)
            en_vi = nest_model.encoder(img_vi)
            # fusion
            f = fusion_model(en_ir, en_vi)
            f = f
            # decoder
            outputs = nest_model.decoder_train(f)
            # save_img(outputs[0],"/root/autodl-tmp/test_for_paper/Code_For_ITCD/")
            # resolution loss: between fusion image and visible image
            x_ir = Variable(img_ir.data.clone(), requires_grad=False)
            x_vi = Variable(img_vi.data.clone(), requires_grad=False)

            ######################### LOSS FUNCTION #########################
            all_ssim_loss = 0.0
            all_fea_loss = 0.0
            all_texture_loss = 0.0
            for output in outputs:
                # output[output>255]=255
                # output[output<0]=0
                # output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
                # output = output * 255
                if count % 50 == 0:
                    save_img(
                        output, "/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved"
                    )
                # ---------------------- LOSS IMAGES ------------------------------------
                # detail loss
                ssim_loss_temp_vi = vi_ssim_loss(output, x_vi)
                ssim_loss_temp_ir = ir_ssim_loss(output, x_ir)
                loss1_value += ssim_vi * (1 - ssim_loss_temp_vi) + ssim_ir * (
                    1 - ssim_loss_temp_ir
                )

                # # feature loss
                g2_ir_fea = en_ir
                g2_vi_fea = en_vi
                g2_fuse_fea = f

                w_ir = [w1]*4
                w_vi = [w2]*4
                w_fea = [1, 10, 100, 1000]
                for ii in range(4):
                    g2_ir_temp = g2_ir_fea[ii]
                    g2_vi_temp = g2_vi_fea[ii]
                    g2_fuse_temp = g2_fuse_fea[ii]
                    (bt, cht, ht, wt) = g2_ir_temp.size()
                    loss2_value += w_fea[ii] * mse_loss(
                        g2_fuse_temp, w_ir[ii] * g2_ir_temp + w_vi[ii] * g2_vi_temp
                    )
                # loss2_value += mse_loss(output,w1*x_vi+w2*x_ir)
                gradinet_loss = F.l1_loss(
                    gradient(output), torch.max(gradient(x_ir), gradient(x_vi))
                )
            loss1_value /= len(outputs)
            loss2_value /= len(outputs)
            gradinet_loss /= len(output)

            total_loss = beta * loss1_value + 0.07*loss2_value + gamma * gradinet_loss
            iter1.set_postfix(
                count=count,
                loss_mse=loss2_value.item(),
                loss_texture=gamma * gradinet_loss.item(),
                loss_ssim=beta * loss1_value.item(),
            )
            total_loss.backward()
            optimizer.step()

            all_fea_loss += loss2_value.item()
            all_ssim_loss += beta * loss1_value.item()
            all_texture_loss += gamma * gradinet_loss.item()
            if index % 30 == 0:
                tensor_writer.add_scalar("Training/true_ssim_loss", loss1_value.item(), count)
                tensor_writer.add_scalar("Training/true_mse_loss", loss2_value.item(), count)
                tensor_writer.add_scalar(
                    "Training/true_texture_loss", gradinet_loss.item(), count
                )
                tensor_writer.add_scalar("Training/ssim_loss", all_ssim_loss, count)
                # tensor_writer.add_scalar("Training/true_mse_loss", loss2_value.item(), count)
                tensor_writer.add_scalar(
                    "Training/texture_loss", all_texture_loss, count
                )
                tensor_writer.add_scalar(
                    "Training/total_loss", total_loss.item(), count
                )
                # tensor_writer.add_scalar("Training/ssim",all_ssim_loss,index+e*len(dataloader))
                # tensor_writer.add_scalar("Training/ms_ssim",all_fea_loss,index+e*len(dataloader))
                # tensor_writer.add_scalar("Training/total_loss",total_loss.item(),index+e*len(dataloader))
            if index % 700 == 0:
                ssim, ms_ssim = evaluate1(
                    nest_model,
                    fusion_model,
                    "kaist",
                    r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/dataset/test/kaist",
                    beta,
                    gamma,
                    w1,
                    w2,
                    ssim_vi,
                    ssim_ir,
                    tensor_writer,
                    count,
                    root_dir,
                )
                nest_model.train()
                fusion_model.train()
                if ssim > max_ssim or ms_ssim > max_ms_ssim:
                    _save_checkpoint(fusion_model, root_dir, "fusion model")
                    _save_checkpoint(nest_model, root_dir, "nest model")
                    path = str(
                        os.path.join(root_dir, f"{index+e*len(dataloader)}次融合")
                    )
                    output = outputs[0]
                    # output[output > 255] = 255
                    # output[output < 0] = 0
                    # output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
                    # output = output * 255
                    os.makedirs(path, exist_ok=True)
                    save_img(output, path)
                    path = str(
                        os.path.join(root_dir, f"{index+e*len(dataloader)}次视觉")
                    )
                    os.makedirs(path, exist_ok=True)
                    save_img(img_vi, path)
                    path = str(
                        os.path.join(root_dir, f"{index+e*len(dataloader)}次热成像")
                    )
                    os.makedirs(path, exist_ok=True)
                    save_img(img_ir, path)
            count += 1
            # if (batch + 1) % args.log_interval == 0:
            # 	mesg = "{}\t Alpha: {} \tW-IR: {}\tEpoch {}:\t[{}/{}]\t ssim loss: {:.6f}\t fea loss: {:.6f}\t total: {:.6f}".format(
            # 		time.ctime(), alpha, w1, e + 1, count, batches,
            # 					  all_ssim_loss / args.log_interval,
            # 					  all_fea_loss / args.log_interval,
            # 					  (all_fea_loss + all_ssim_loss) / args.log_interval
            # 	)
            # 	tbar.set_description(mesg)
            # 	Loss_ssim.append( all_ssim_loss / args.log_interval)
            # 	Loss_feature.append(all_fea_loss / args.log_interval)
            # 	Loss_all.append((all_fea_loss + all_ssim_loss) / args.log_interval)
            # 	count_loss = count_loss + 1
            # 	all_ssim_loss = 0.
            # 	all_fea_loss = 0.

        # save model
        # save_model_filename = mode + ".model"
        # save_model_path = os.path.join(temp_path_model_w, save_model_filename)
        # torch.save(fusion_model.state_dict(), save_model_path)

        # print("\nDone, trained model saved at", save_model_path)
    tensor_writer.close()


if __name__ == "__main__":
    main()
