import torch

from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


def window_partition(x, window_size):

    B, H, W, C = x.shape

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip =  inp // reduction

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class Convlutioanl(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Convlutioanl, self).__init__()
        self.padding=(1,1,1,1)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=0,stride=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, input):
        out=F.pad(input,self.padding,'replicate')
        out=self.conv(out)
        out=self.bn(out)
        out=self.relu(out)
        return out


class Convlutioanl_out(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Convlutioanl_out, self).__init__()

        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,stride=1)

        self.sigmoid=nn.Sigmoid()

    def forward(self, input):

        out=self.conv(input)

        out=self.sigmoid(out)
        return out



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EqualAtt(nn.Module):
    def __init__(self, dim):
        super( EqualAtt, self).__init__()
        self.key_embed = Convlutioanl(dim,dim)

        factor = 8
        self.embed = nn.Sequential(
            nn.Conv2d( dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, 1, kernel_size=1),
            nn.BatchNorm2d(1)

        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )


        self.bn = nn.BatchNorm2d(dim)
        self.relu= nn.ReLU(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self,  v,k,q):
        k = self.key_embed(k)
        q = self.key_embed(q)
        qk = q+k

        w = self.embed(qk)

        v = self.conv1x1(v)

        mul=w*v


        return  mul

class CotLayer(nn.Module):
    def __init__(self, dim):
        super(CotLayer, self).__init__()
        self.key_embed = Convlutioanl(dim,dim)

        factor = 8
        self.embed = nn.Sequential(
            nn.Conv2d( dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, 1, kernel_size=1),
            nn.BatchNorm2d(1)

        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )


        self.bn = nn.BatchNorm2d(dim)
        self.relu= nn.ReLU(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self,  v,k,q,):
        k = self.key_embed(k)
        qk = q+k

        w = self.embed(qk)

        v = self.conv1x1(v)

        mul=w*v
        out= mul+  k


        return out


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape

        A=self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):

        flops = 0

        flops += N * self.dim * 3 * self.dim

        flops += self.num_heads * N * (self.dim // self.num_heads) * N

        flops += self.num_heads * N * N * (self.dim // self.num_heads)

        flops += N * self.dim * self.dim
        return flops



class SwinTransformerBlock(nn.Module):


    def __init__(self, dim, input_resolution, num_heads, window_size=1, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:

            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):

        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):

        B,C,H,W= x.shape

        x=x.view(B,H,W,C)
        shortcut = x
        shape=x.view(H*W*B,C)
        x = self.norm1(shape)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x


        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        B,H,W,C=x.shape
        x=x.view(B,C,H,W)


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += self.dim * H * W

        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)

        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio

        flops += self.dim * H * W
        return flops


class PatchEmbed(nn.Module):


    def __init__(self, img_size=120, patch_size=4, in_chans=6, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])


        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:

            x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class MODEL(nn.Module):
    def __init__(self, in_channel=1, channel16=16, channel2=2,channel32=32, out_channel=64,
                 output_channel=1, dim=32, mlp_ratio=4.,  drop=0.,  depth=3, num_heads=8, window_size=1,
                 img_size=120, patch_size=4, embed_dim=96,
                 attn_drop=0.,drop_path=0.,
                 qkv_bias=True, qk_scale=None,downsample=None,patch_norm=True,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,use_checkpoint=False):
        super(MODEL, self).__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.conv16= Convlutioanl(in_channel,channel16)
        self.conv1_32 = Convlutioanl(in_channel, channel32)
        self.conv32 = Convlutioanl(channel32, channel32)
        self.conv16_32 = Convlutioanl(channel16, channel32)
        self.conv16_16 = Convlutioanl(channel16, channel16)
        self.conv2_16 = Convlutioanl(channel2, channel16)
        self.conv2_32 = Convlutioanl(channel2, channel32)
        self.layer2 = Convlutioanl(out_channel, out_channel)
        self.convolutional_out =  Convlutioanl_out(channel16,output_channel)
        self.n_feats = 16
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.CoordAtt= CoordAtt(channel16,channel16)
        self.TS_lv1 = CotLayer(dim=16)
        self.equalAtt =EqualAtt(dim=16)
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.basicLayer = BasicLayer(dim=channel16,
                                     input_resolution=(patches_resolution[0], patches_resolution[1]),
                                     depth=depth,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     downsample=downsample,
                                     use_checkpoint=use_checkpoint)
    def forward(self, ir,vi):

        layerA1_1 = self.conv16(ir)
        layerA1_2 = self.conv16_16(layerA1_1)
        layerA1_3 = self.conv16_16(layerA1_2)
        resA1=layerA1_1+layerA1_3
        Q_A1=resA1
        K_B1 = resA1

        layerB1_1 = self.conv16(vi)
        layerB1_2= self.conv16_16(layerB1_1)
        layerB1_3 = self.conv16_16(layerB1_2)
        resB1 = layerB1_1 + layerB1_3
        Q_B1=resB1
        K_A1 = resB1

        cat=torch.cat((ir , vi ), 1)
        conv1_1=self.conv2_16( cat)
        conv1_2 = self.conv16_16(conv1_1)

        coorAtt1_1=self.CoordAtt( conv1_2)
        coorAtt1_2 = self.CoordAtt( coorAtt1_1)
        nlb1= conv1_2+coorAtt1_2
        v1= nlb1

        attentionA1= self.TS_lv1(v1,K_A1,  Q_A1)
        attentionB1 = self.TS_lv1(v1, K_B1, Q_B1)

        layerA2_1=self.conv16_16( attentionA1 )
        layerA2_2 = self.conv16_16( layerA2_1)
        layerA2_3 = self.conv16_16(layerA2_2)
        resA2 = layerA2_1 + layerA2_3

        Q_A2= resA2
        K_B2= resA2

        layerB2_1 = self.conv16_16( attentionB1 )
        layerB2_2 = self.conv16_16(layerB2_1)
        layerB2_3 = self.conv16_16(layerB2_2)
        resB2= layerB2_1 + layerB2_3
        Q_B2= resB2
        K_A2= resB2

        conv2_1 = self.conv16_16( v1)
        conv2_2= self.conv16_16(conv2_1)
        coorAtt2_1 = self.CoordAtt(conv2_2)
        coorAtt2_2 = self.CoordAtt(coorAtt2_1)
        nlb2 = conv2_2 + coorAtt2_2
        v2=nlb2


        attentionA2 = self.TS_lv1(v2, K_A2, Q_A2)
        attentionB2 = self.TS_lv1(v2, K_B2, Q_B2)

        layerA3_1 = self.conv16_16( attentionA2)
        layerA3_2 = self.conv16_16( layerA3_1 )
        layerA3_3 = self.conv16_16(layerA3_2)
        resA3 = layerA3_1 + layerA3_3

        Q_A3 = resA3
        K_B3 = resA3

        conv3_1 = self.conv16_16(v2)
        conv3_2 = self.conv16_16(conv3_1)
        coorAtt3_1 = self.CoordAtt(conv3_2)
        coorAtt3_2 = self.CoordAtt(coorAtt3_1)
        nlb3 = conv3_2 + coorAtt3_2
        v3 = nlb3

        layerB3_1 = self.conv16_16( attentionB2)
        layerB3_2= self.conv16_16(layerB3_1)
        layerB3_3 = self.conv16_16(layerB3_2)
        resB3 = layerB3_1 + layerB3_3

        Q_B3 =  resB3
        K_A3 = resB3


        attentionA3 = self.TS_lv1(v3, K_A3, Q_A3)
        attentionB3 = self.TS_lv1(v3, K_B3, Q_B3)

        att= self.equalAtt(v3, attentionA3,attentionB3)
        coorAtt4_1 = self.CoordAtt(att)
        coorAtt4_2 = self.CoordAtt(coorAtt4_1)
        nlb4 =  att + coorAtt4_2

        encode_size_DTRM1 = (nlb4.shape[2],nlb4.shape[3])
        swinTransformer_DTRM1 = self.basicLayer(nlb4 , encode_size_DTRM1)

        out1 = self.conv16_16(swinTransformer_DTRM1 )
        out2 =self.convolutional_out (out1)
        return out2



