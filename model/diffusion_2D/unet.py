import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
    

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=4, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)
    

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(4, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx
        
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input
        
        
#SelfAttention_fuse adapted to 2D
class SelfAttention_fuse(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(4, in_channel)
        self.out = nn.Conv2d(in_channel, in_channel, 1)
        self.defmgen=nn.Conv2d(in_channel,3,3,padding=1)
        self.nonlinear=nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, q,k,v,size):
        batch, channel, height, width = q.shape

        n_head = self.n_head
        residual=q
        norm_q = self.norm(q)
        norm_k = self.norm(k)
        norm_v = self.norm(v)
        

        qkv=torch.cat([norm_q,norm_k,norm_v],dim=1)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=n_head, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=n_head, h=height, w=width)
        out = self.out(out)
        out=self.defmgen(out+residual)
        out=F.upsample_nearest(out,size)
        out=self.nonlinear(out)
        return out
    

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x
    

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128,
        opt=None
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None
        self.opt=opt
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               time_emb_dim=time_dim, dropout=dropout, with_attn=False)
        ])


        ups_diff = []
        ups_regis = []
        ups_adapt=[]
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                feat_channel=feat_channels.pop()
                ups_diff.append(ResnetBlocWithAttn(
                    pre_channel+feat_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                regischannel=pre_channel+feat_channel+channel_mult
                ups_regis.append(ResnetBlocWithAttn(
                    regischannel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                ups_adapt.append(
                    SelfAttention_fuse(channel_mult)  #################################################
                )
                pre_channel = channel_mult
            if not is_last:
                ups_adapt.append(nn.Identity())
                ups_diff.append(Upsample(pre_channel))
                ups_regis.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups_diff = nn.ModuleList(ups_diff)
        self.ups_regis = nn.ModuleList(ups_regis)
        self.ups_adapt=nn.ModuleList(ups_adapt)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel))
        # self.final_attn=SelfAttention_fuse(1)
        self.final_conv_defm = Block(pre_channel+1,3,groups=3)
        #print('------------------pre channel------------------------')
        #print(pre_channel)
        #self.final_conv_defm = Block(pre_channel,2,groups=32)    ################################!!!!!!!!Em dúvida!!!!!!!


    def forward(self, x,x_m, time):
        #print('--------x size----------------')
        #print(x.size())
        input_size=(x.size(2),x.size(3))   ################ entrada unet: in_channel
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        x_1=x
        x_2=x
        defm=[]
        # x_1vis=[]
        for layerd,layerr,layera in zip(self.ups_diff,self.ups_regis,self.ups_adapt):
            if isinstance(layerd, ResnetBlocWithAttn):
                feat=feats.pop()
                x_1 = layerd(torch.cat((x_1, feat), dim=1), t)
                x_2 = layerr(torch.cat((x_2, feat,x_1), dim=1), t)
                defm_=layera(x_2,x_1,x_1,input_size)     ###############################################
                defm.append(defm_)
            else:
                x_1 = layerd(x_1)
                x_2 = layerr(x_2)
        recon=self.final_conv(x_1)
        defm=torch.stack(defm,dim=1)
        defm=torch.cat([defm,self.final_conv_defm(torch.cat((x_2,recon), dim=1)).unsqueeze_(1)],dim=1)
        defm=torch.mean(defm,dim=1)
        return recon,defm
    
class Dense2DSpatialTransformer(nn.Module):
    def __init__(self):
        super(Dense2DSpatialTransformer, self).__init__()

    def forward(self, input1, input2):
        #input1 = (input1[:, :1]+1)/2.0
        return self._transform(input1, input2[:,0], input2[:,1])
        
    def _transform(self, input1, dHeight, dWidth):
            batchSize = dHeight.shape[0]

            hgt = dHeight.shape[1]
            wdt = dHeight.shape[2]

            H_mesh, W_mesh = self._meshgrid(hgt, wdt)

            H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, hgt, wdt)
            W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, hgt, wdt)

            H_upmesh = dHeight + H_mesh
            W_upmesh = dWidth + W_mesh
            
            return self._interpolate(input1, H_upmesh, W_upmesh)
        
    def _meshgrid(self, hgt, wdt):
        h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()

        w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()

        return h_t, w_t
    
    def _interpolate(self, input, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch    = input.shape[1]
        height = input.shape[2]
        width  = input.shape[3]

        img = torch.zeros(nbatch, nch, height+2, width+2).cuda()
        img[:, :, 1:-1, 1:-1] = input
        img[:, :, 0, 1:-1] = input[:, :, 0, :]
        img[:, :, -1, 1:-1] = input[:, :, -1, :]
        img[:, :, 1:-1, 0] = input[:, :, :, 0]
        img[:, :, 1:-1, -1] = input[:, :, :, -1]
        img[:, :, 0, 0] = input[:, :, 0, 0]
        img[:, :, 0, -1] = input[:, :, 0, -1]
        img[:, :, -1, 0] = input[:, :, -1, 0]
        img[:, :, -1, -1] = input[:, :,-1, -1]

        imgHgt = img.shape[2]
        imgWdt = img.shape[3]

        # H_upmesh, W_upmesh = [H, W] -> [BHW,]
        H_upmesh = H_upmesh.view(-1).float()+1.0  # (BHW,)
        W_upmesh = W_upmesh.view(-1).float()+1.0  # (BHW,)

        # H_upmesh, W_upmesh -> Clamping
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        hf = torch.clamp(hf, 0, imgHgt-1)  # (BHW,)
        hc = torch.clamp(hc, 0, imgHgt-1)  # (BHW,)
        wf = torch.clamp(wf, 0, imgWdt-1)  # (BHW,)
        wc = torch.clamp(wc, 0, imgWdt-1)  # (BHW,)

        # Find batch indexes
        rep = torch.ones([height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
        bHW = torch.matmul((torch.arange(0, nbatch).float()*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

        # Box updated indexes
        W = imgWdt
        # x: W, y: H, z: D
        idx_00 = bHW + hf*W + wf
        idx_10 = bHW + hf*W + wc
        idx_01 = bHW + hc*W + wf
        idx_11 = bHW + hc*W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_00 = torch.index_select(img_flat, 0, idx_00.long())
        val_10 = torch.index_select(img_flat, 0, idx_10.long())
        val_01 = torch.index_select(img_flat, 0, idx_01.long())
        val_11 = torch.index_select(img_flat, 0, idx_11.long())

        dHeight = hc.float() - H_upmesh
        dWidth  = wc.float() - W_upmesh

        wgt_00 = (dHeight*dWidth).unsqueeze_(1)
        wgt_10 = (dHeight * (1-dWidth)).unsqueeze_(1)
        wgt_01 = ((1-dHeight) * dWidth).unsqueeze_(1)
        wgt_11 = ((1-dWidth) * (1-dHeight)).unsqueeze_(1)

        output = val_00*wgt_00 + val_10*wgt_10 + val_01*wgt_01 + val_11*wgt_11
        output = output.view(nbatch, height, width, nch).permute(0, 3, 1, 2)  #B, C, H, W
        return output