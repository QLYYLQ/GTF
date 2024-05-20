import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from pytorch_msssim import MS_SSIM,ssim
import random
from dataloader import get_dataset,get_datapath
from model.net import NestFuse_light2_nodense,Fusion_network
from tqdm import tqdm,trange
import os
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
import cv2
import gc
epochs = 3
EPSILON=1e-5

def _save_checkpoint(model, dir, save_best=False, overwrite=True):
    checkpoint = {}
    checkpoint['model'] = model.state_dict()

    filename = os.path.join(dir,"best.pth")
    torch.save(checkpoint, filename)
    print("Saving current best model: best_model.pth")
    

def evaluate(nest_model,fusion_model,dataset_name,dataset_path,alpha,w_vi,w_ir,tensorboard_writer,save_dir=None):
    dataset = get_dataset(dataset_name)(dataset_path)
    dataloader = DataLoader(dataset,batch_size=6,num_workers=4,drop_last=True)
    ssim_loss = MS_SSIM(data_range=1.0,channel=1)
    mse_loss = torch.nn.MSELoss()
    nest_model.eval()
    fusion_model.eval()
    ssim_list = []
    ms_ssim_list = []
    for index,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        img_vi ,img_ir = batch
        img_vi = img_vi.cuda()
        img_ir = img_ir.cuda()
        loss1_value = 0
        loss2_value = 0
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





def save_img(img,path):
	batch = img.shape[0]
	for i in range(batch):
		img = img.detach()
		img_numpy = img[i].squeeze().cpu().numpy()
		img_numpy = (img_numpy*255).astype(np.uint8)
		img_resize = cv2.resize(img_numpy,(640,512),cv2.INTER_AREA)
		path1 = os.path.join(path,f"第{i}张图.png")
		cv2.imwrite(str(path1),img_resize)


def set_seed(seed=42):
    """为PyTorch、NumPy和Python内置的random库设置种子，以确保实验可重复性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    # 确保PyTorch的卷积操作的确定性，这可能会降低一些运行效率
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 你可以选择任何喜欢的整数作为种子

def create_dir(alpha,w1,w2):
	root_dir = r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/saved"
	path = Path(root_dir)
	dir_path = os.path.join(path,str(alpha)+"--"+"w1:"+str(w1)+"--w2:"+str(w2))
	dir_path = str(dir_path)
	os.makedirs(dir_path,exist_ok=True)
	return dir_path




def main():
	dataset = get_dataset("kaist")(get_datapath("kaist"))
	# True - RGB , False - gray
	img_flag = False
	alpha_list = [700,800,900]
	w_all_list = [[6.0, 3.0],[7.0,3.0],[6.0,4.0]]

	for w_w in w_all_list:
		w1, w2 = w_w
		for alpha in alpha_list:
			train(dataset, img_flag, alpha, w1, w2)


def train(dataset, img_flag, alpha, w1, w2):

	batch_size =6
	# load network model
	nc = 1
	input_nc = nc
	output_nc = nc
	#nb_filter = [64, 128, 256, 512]
	nb_filter = [64, 112, 160, 208]
	f_type = 'res'
	root_dir = create_dir(alpha,w1,w2)
	dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)
	with torch.no_grad():
		deepsupervision = False
		nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
		model_path = r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/model/nestfuse_gray_1e2.model"
		# load auto-encoder network
		print('Resuming, initializing auto-encoder using weight from {}.'.format(model_path))
		nest_model.load_state_dict(torch.load(model_path))
		nest_model.cuda()
		nest_model.eval()

	# fusion network
	fusion_model = Fusion_network(nb_filter, f_type)
	fusion_model.cuda()
	fusion_model.train()

	# if args.resume_fusion_model is not None:
	# 	print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_fusion_model))
	# 	fusion_model.load_state_dict(torch.load(args.resume_fusion_model))
	optimizer = torch.optim.Adam(fusion_model.parameters(), 3e-4)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = MS_SSIM(data_range=1.0,channel=1)


	tbar = trange(epochs)
	print('Start training.....')
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

	Loss_feature = []
	Loss_ssim = []
	Loss_all = []
	count_loss = 0
	
	nest_model.cuda()
	#trans_model.cuda()
	fusion_model.cuda()
	sobel_loss = nn.L1Loss()
	tensor_writer = SummaryWriter(root_dir,flush_secs=30)
	max_ssim=0
	max_ms_ssim = 0
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		
		count = 0
		for index,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
			ssim=0
			ms_ssim = 0
			all_ssim_loss = 0.
			all_fea_loss = 0.
			loss1_value=0.
			loss2_value=0.
			img_vi,img_ir=batch
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()
			# encoder
			en_ir = nest_model.encoder(img_ir)
			en_vi = nest_model.encoder(img_vi)
			# fusion
			f = fusion_model(en_ir, en_vi)
			# decoder
			outputs = nest_model.decoder_eval(f)
			# save_img(outputs[0],"/root/autodl-tmp/test_for_paper/Code_For_ITCD/")
			# resolution loss: between fusion image and visible image
			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)

			######################### LOSS FUNCTION #########################
			all_ssim_loss = 0.
			all_fea_loss = 0.
			for output in outputs:
				output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
				output = output * 255
				# ---------------------- LOSS IMAGES ------------------------------------
				# detail loss
				ssim_loss_temp2 = ssim_loss(output, x_vi)
				loss1_value += alpha * (1 - ssim_loss_temp2)

				# feature loss
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
					loss2_value += w_fea[ii]*mse_loss(g2_fuse_temp, w_ir[ii]*g2_ir_temp + w_vi[ii]*g2_vi_temp)

			loss1_value /= len(outputs)
			loss2_value /= len(outputs)
			

			
			total_loss = loss1_value + loss2_value 
			total_loss.backward()
			optimizer.step()

			all_fea_loss += loss2_value.item() # 
			all_ssim_loss += loss1_value.item() # 
			if index %100 == 0:
				tensor_writer.add_scalar("Training/ssim_loss",all_ssim_loss,index+e*len(dataset))
				tensor_writer.add_scalar("Training/mse_loss",all_fea_loss,index+e*len(dataset))
				tensor_writer.add_scalar("Training/total_loss",total_loss.item(),index+e*len(dataset))
			if index%400 ==0:
				ssim,ms_ssim = evaluate(nest_model,fusion_model,"kaist",r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/dataset/test/kaist_wash_picture_test",alpha,w_vi,w_ir,tensor_writer,root_dir)
				if ssim>max_ssim or ms_ssim > max_ms_ssim:
					_save_checkpoint(fusion_model,root_dir)
					path = str(os.path.join(root_dir,f"{index+e*len(dataset)}"))
					os.makedirs(path,exist_ok=True)
					save_img(outputs[0],path)
				
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
