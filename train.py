import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from pytorch_msssim import MS_SSIM

from dataloader import get_dataset,get_datapath
from model.net import NestFuse_light2_nodense,Fusion_network
from tqdm import tqdm,trange
import os

from torch.autograd import Variable
import cv2
epochs = 3
EPSILON=1e-5



def save_img(img,path):
	batch = img.shape[0]
	for i in range(batch):
		img = img.detach()
		img_numpy = img[i].squeeze().cpu().numpy()
		img_numpy = (img_numpy*255).astype(np.uint8)
		img_resize = cv2.resize(img_numpy,(640,512),cv2.INTER_AREA)
		cv2.imwrite(path+f"第{i}张图.png",img_resize)
		break






def main():
	dataset = get_dataset("kaist")(get_datapath("kaist"))
	# True - RGB , False - gray
	img_flag = False
	alpha_list = [700]
	w_all_list = [[6.0, 3.0]]

	for w_w in w_all_list:
		w1, w2 = w_w
		for alpha in alpha_list:
			train(dataset, img_flag, alpha, w1, w2)


def train(dataset, img_flag, alpha, w1, w2):

	batch_size =4
	# load network model
	nc = 1
	input_nc = nc
	output_nc = nc
	#nb_filter = [64, 128, 256, 512]
	nb_filter = [64, 112, 160, 208]
	f_type = 'res'

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
	all_ssim_loss = 0.
	all_fea_loss = 0.
	sobel_loss = nn.L1Loss()
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		
		count = 0
		nest_model.cuda()
		#trans_model.cuda()
		fusion_model.cuda()
		for batch in tqdm(dataloader,total=len(dataloader)):
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
			save_img(outputs[0],"/root/autodl-tmp/test_for_paper/Code_For_ITCD/")
			# resolution loss: between fusion image and visible image
			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)

			######################### LOSS FUNCTION #########################
			loss1_value = 0.
			loss2_value = 0.
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
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t Alpha: {} \tW-IR: {}\tEpoch {}:\t[{}/{}]\t ssim loss: {:.6f}\t fea loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), alpha, w1, e + 1, count, batches,
								  all_ssim_loss / args.log_interval,
								  all_fea_loss / args.log_interval,
								  (all_fea_loss + all_ssim_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_ssim.append( all_ssim_loss / args.log_interval)
				Loss_feature.append(all_fea_loss / args.log_interval)
				Loss_all.append((all_fea_loss + all_ssim_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_fea_loss = 0.

		# save model
		save_model_filename = mode + ".model"
		save_model_path = os.path.join(temp_path_model_w, save_model_filename)
		torch.save(fusion_model.state_dict(), save_model_path)

		print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)


if __name__ == "__main__":
	main()
