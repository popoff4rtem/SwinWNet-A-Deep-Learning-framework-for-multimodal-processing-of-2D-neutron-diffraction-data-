import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_chans=1, embed_dim=48):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Padding if necessary to make H and W divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
            new_H, new_W = H + pad_h, W + pad_w
        else:
            new_H, new_W = H, W
        
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x, (new_H, new_W)  # Return padded sizes for later use


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp_, Wp_): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_reverse(windows, window_size, H, W):
    """
    Reverse windows back to original shape.
    Args:
        windows (tensor): [B * num_windows, window_size, window_size, C]
        window_size (int)
        H, W (int): original height and width
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def compute_mask(H, W, window_size, shift_size, device):
    """
    Compute attention mask for shifted windows.
    """
    # Pad feature maps to multiples of window size
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    img_mask = torch.zeros((1, H + pad_b, W + pad_r, 1), device=device)  # 1 Hp Wp 1
    
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    
    # Partition mask into windows
    mask_windows, _ = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    
    # Compute relative mask
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(0)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1).long()  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
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

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=5, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)  # Simplified, can add stochastic depth later
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, resolution):
        B, L, C = x.shape
        H, W = resolution
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = compute_mask(H, W, self.window_size, self.shift_size, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows, (Hp, Wp) = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Handle padding if any
        if Hp != H or Wp != W:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, resolution):
        B, L, C = x.shape
        H, W = resolution
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        # Pad if H or W is odd
        pad_h = 1 if H % 2 == 1 else 0
        pad_w = 1 if W % 2 == 1 else 0
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            new_H, new_W = H + pad_h, W + pad_w
        else:
            new_H, new_W = H, W

        # Merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B (H/2*W/2) 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B (H/2*W/2) 2*C

        new_resolution = (new_H // 2, new_W // 2)
        return x, new_resolution
    
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=5, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,  # <--- всегда 0, как в рабочем скрипте
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path
            ) for _ in range(depth)
        ])

    def forward(self, x, resolution):
        for blk in self.blocks:
            x = blk(x, resolution)
        return x

class SwinEncoder(nn.Module):
    def __init__(self, patch_size=2, in_chans=1, embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=5, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        self.layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        dim = embed_dim
        for i in range(len(depths) - 1):
            self.layers.append(BasicLayer(dim=dim, depth=depths[i], num_heads=num_heads[i],
                                          window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                          drop=drop, attn_drop=attn_drop, drop_path=drop_path))
            self.downs.append(PatchMerging(dim=dim))
            dim *= 2
        
        self.layers.append(BasicLayer(dim=dim, depth=depths[-1], num_heads=num_heads[-1],
                                      window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      drop=drop, attn_drop=attn_drop, drop_path=drop_path))
    
    def forward(self, x):
        skips = []
        res_skips = []
        x, padded_img_size = self.patch_embed(x)
        resolution = (padded_img_size[0] // self.patch_embed.patch_size, padded_img_size[1] // self.patch_embed.patch_size)
        
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, resolution)
            skips.append(x)
            res_skips.append(resolution)
            x, resolution = self.downs[i](x, resolution)
        
        x = self.layers[-1](x, resolution)
        skips.append(x)
        res_skips.append(resolution)
        
        return skips, res_skips, resolution, padded_img_size  # now return res_skips too
    
class Bottleneck(nn.Module):
    def __init__(self, dim, num_heads, window_size=5, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.layer = BasicLayer(dim=dim, depth=2, num_heads=num_heads,
                                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                drop=drop, attn_drop=attn_drop, drop_path=drop_path)
    
    def forward(self, x, resolution):
        return self.layer(x, resolution)
    
class PatchExpanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x, resolution):
        B, L, C = x.shape
        H, W = resolution
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)  # [B, L, 2*C]
        x = x.view(B, H, W, 2 * C)
        x = x.reshape(B, H, W, 2, 2, C // 2)  # [B, H, W, 2, 2, C//2]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, 2*H, 2*W, C // 2)

        new_resolution = (2 * H, 2 * W)

        x = x.view(B, -1, C // 2)
        x = self.norm(x)

        return x, new_resolution
    
def crop_to_res(x, current_res, target_res):
    B, L, C = x.shape
    current_H, current_W = current_res
    target_H, target_W = target_res

    assert current_H >= target_H and current_W >= target_W

    x = x.view(B, current_H, current_W, C)
    x = x[:, :target_H, :target_W, :]
    x = x.reshape(B, target_H * target_W, C)
    return x


class SwinDecoder(nn.Module):
    def __init__(self, embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=5, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()

        self.ups = nn.ModuleList()
        self.swin_blocks = nn.ModuleList()
        self.linears = nn.ModuleList()

        dim = embed_dim * 8    # 384
        self.depths = depths[-2::-1]
        self.num_heads = num_heads[-2::-1]

        for i in range(len(depths) - 1):

            # expand: dim → dim/2
            expanded_dim = dim // 2

            # concat with skip (same expanded_dim)
            concat_dim = expanded_dim * 2   # = dim

            self.ups.append(PatchExpanding(dim=dim))

            self.swin_blocks.append(
                BasicLayer(
                    dim=concat_dim,
                    depth=self.depths[i],
                    num_heads=self.num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                )
            )

            self.linears.append(nn.Linear(concat_dim, expanded_dim))

            dim = expanded_dim  # prepare for next iteration

    def forward(self, x, resolution, skips, skip_res_list):
        skips = skips[-2::-1]
        skip_res_list = skip_res_list[-2::-1]

        for i in range(len(self.swin_blocks)):

            # expand
            x, new_res = self.ups[i](x, resolution)

            # crop to skip resolution
            target_res = skip_res_list[i]
            if new_res != target_res:
                x = crop_to_res(x, new_res, target_res)

            # concat skip
            x = torch.cat([x, skips[i]], dim=-1)

            # swin processing
            x = self.swin_blocks[i](x, target_res)

            # reduce channels
            x = self.linears[i](x)

            resolution = target_res

        return x, resolution
    
class SegmentationHead(nn.Module):
    def __init__(self, embed_dim=48, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        
        # simple but effective segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1)
        )

    def forward(self, x, resolution):
        """
        x: [B, N, C]
        resolution: (H/2, W/2) padded sizes before patching
        """
        B, N, C = x.shape
        H, W = resolution
        
        # compute patch resolution
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        # reshape back to image-like structure
        x = x.transpose(1, 2).reshape(B, C, H_patch, W_patch)

        # small convolutional decoder
        x = self.seg_head(x)  # [B, 1, H_patch, W_patch]

        # upsample to original padded resolution
        x = nn.functional.interpolate(x, scale_factor=self.patch_size, mode='bilinear')

        # crop to remove padding (if any)
        x = x[:, :, :H, :W]

        return x  # [B, 1, H, W]

class SwinUNet(nn.Module):
    def __init__(self, patch_size=2, in_chans=1, embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=5, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        # self.patch_embed = PatchEmbed(patch_size, 1, embed_dim)

        self.encoder = SwinEncoder(patch_size=patch_size, 
                    in_chans=in_chans, 
                    embed_dim=embed_dim, 
                    depths=depths, 
                    num_heads=num_heads, 
                    window_size=window_size,
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=True, 
                    drop=drop, 
                    attn_drop=attn_drop, 
                    drop_path=drop_path
                    )
        self.bottleneck = Bottleneck(dim=embed_dim*8, 
                    num_heads=num_heads[-1], 
                    window_size=window_size)
        
        self.decoder = SwinDecoder(embed_dim=embed_dim, 
                    depths=depths, 
                    num_heads=num_heads,
                    window_size=window_size, 
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias, 
                    drop=drop, 
                    attn_drop=attn_drop, 
                    drop_path=drop_path)
        
        self.head = SegmentationHead(embed_dim=embed_dim, 
                    patch_size=patch_size)

    def forward(self, x):
        # embed
        # x_patch, padded_res = self.patch_embed(x)

        # encode
        skips, skip_res_list, bott_res, padded_img_size = self.encoder(x)

        # bottleneck
        x_bottleneck = self.bottleneck(skips[-1], bott_res)

        # decode
        x_dec, dec_res = self.decoder(x_bottleneck, bott_res, skips, skip_res_list)

        # segmentation map
        seg = self.head(x_dec, padded_img_size)

        return seg
