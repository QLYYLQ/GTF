from dataloader import get_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

EPSILON=1e-5

from pytorch_msssim import MS_SSIM,ssim
from torch.autograd import Variable


def evaluate(nest_model,fusion_model,dataset_name,dataset_path,alpha,w_vi,w_ir,tensorboard_writer,save_dir=None):
    dataset = get_dataset(dataset_name)(dataset_path)
    dataloader = DataLoader(dataset,batch_size=8,num_workers=4,drop_last=True)
    ssim_loss = MS_SSIM(data_range=1.0,channel=1)
    mse_loss = torch.nn.MSELoss()
    nest_model.eval()
    fusion_model.eval()
    for index,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        img_vi ,img_ir = batch
        img_vi = img_vi.cuda()
        img_ir = img_ir.cuda()
        loss1_value = 0
        loss2_value = 0
        ssim_list = []
        ms_ssim_list = []
        with torch.no_grad():
            en_ir = nest_model.encoder(img_ir)
            en_vi = nest_model.encoder(img_vi)
			# fusion
            f = fusion_model(en_ir, en_vi)
			# decoder
            outputs = nest_model.decoder_eval(f)
            x_vi = Variable(img_vi.data.clone(), requires_grad=False)
            for output in outputs:
                output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
                output = output * 255
				# ---------------------- LOSS IMAGES ------------------------------------
				# detail loss
                ssim_loss_temp2 = ssim_loss(output, x_vi)
                ssim1 = ssim(output,x_vi,data_range=1.0,size_average=True)
                loss1_value = alpha * (1 - ssim_loss_temp2)
				# feature loss
                g2_ir_fea = en_ir
                g2_vi_fea = en_vi
                g2_fuse_fea = f

                w_fea = [1, 10, 100, 1000]
                for ii in range(4):
                    g2_ir_temp = g2_ir_fea[ii]
                    g2_vi_temp = g2_vi_fea[ii]
                    g2_fuse_temp = g2_fuse_fea[ii]
                    (bt, cht, ht, wt) = g2_ir_temp.size()
                    loss2_value += w_fea[ii]*mse_loss(g2_fuse_temp, w_ir[ii]*g2_ir_temp + w_vi[ii]*g2_vi_temp)

            loss1_value /= len(outputs)
            loss2_value /= len(outputs)
            loss1_value = loss1_value.item()
            loss2_value = loss2_value.item()
            total_loss = loss1_value+loss2_value
            if index%100 == 0:
                tensorboard_writer.add_scalar("Validation/total_loss",total_loss,index)
                tensorboard_writer.add_scalar("Validation/ms_ssim_loss",loss1_value,index)
                tensorboard_writer.add_scalar("Validation/mse_loss",loss2_value,index)
                tensorboard_writer.add_scalar("Validation/ms_ssim",ssim_loss_temp2.item(),index)
                tensorboard_writer.add_scalar("Validation/ssim",ssim1.item(),index)
                ssim_list.append(ssim1.item())
                ms_ssim_list.append(ssim_loss_temp2.item())
    return max(ssim_list),max(ms_ssim_list)