import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

path = r"/root/autodl-tmp/test_for_paper/Code_For_ITCD/model/nestfuse_gray_1e2.model"
from model.net import NestFuse_light2_nodense,Fusion_network
from thop import profile,clever_format
import torch
with torch.no_grad():
    nb_filter = [64, 112, 160, 208]
    deepsupervision=False
    nest_model = NestFuse_light2_nodense(nb_filter,1,1,deepsupervision)
    nest_model.load_state_dict(torch.load(path))
    nest_model = nest_model.cuda()
    nest_model = nest_model.eval()
fusion_model = Fusion_network(nb_filter,'res')
fusion_model = fusion_model.cuda()
fusion_model = fusion_model.eval()
class model(torch.nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.net1 = nest_model
        self.net2 = fusion_model
        self.training = False
    def forward(self,vi,ir):
        en_ir  = self.net1.encoder(ir)
        en_vi = self.net1.encoder(vi)
        f = self.net2(en_ir,en_vi)
        outputs = self.net1.decoder_eval(f)
        return outputs
    
model1 = model()
model1 = model1.cuda()
input1 = torch.randn(10,1,256,256)
input2 = torch.randn(10,1,256,256)
input1 = input1.cuda()
input2 = input2.cuda()
flops,params = profile(model1,inputs=(input1,input2))
flops, params = clever_format([flops, params], '%.3f')
print(flops)
print(params)