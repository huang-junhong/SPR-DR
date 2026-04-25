import loss
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Replacement(nn.Module):
    def __init__(self, r_size=(16,16), size=(128,128), roll_prob=0.5):
        super(Replacement, self).__init__()
        self.r_size = r_size
        self.roll_prob = roll_prob

        self.patch_mean = nn.AvgPool2d(kernel_size=r_size, stride=r_size, padding=0)
        self.unfold = nn.Unfold(r_size, 1, 0, r_size)

        self.fold = nn.Fold(output_size=size, kernel_size=r_size, stride=r_size)
        self.fold_roll = nn.Fold(output_size=(int(size[0]-r_size[0]), int(size[1]-r_size[1])), kernel_size=r_size, stride=r_size)

        self.D_LOSS = loss.D_LOSS()

    def _get_patches(self, input):
        n,c,h,w = input.size()
        current = self.unfold(input)
        current = current.permute([0,2,1])
        current = torch.reshape(current,[n,-1,c,self.r_size[0],self.r_size[1]])
        return current

    def _restore_image(self, input):
        n, num_patch, _, _, _ = input.size()
        current = torch.reshape(input, [n, num_patch, -1])
        current = current.permute(0,2,1)
        if not self.isroll:
            current = self.fold(current)
        else:
            current = self.fold_roll(current)
        return current

    def _generat_mask_random(self, n, length):
        all_mask = []
        r_num = int(self.portion * length)
        for i in range(n):
            mask = np.ones(length)
            inplace = random.sample(range(0, length), r_num)
            mask = np.multiply(mask, list(i in inplace for i in range(length)))
            mask = np.reshape(mask,[1,length,1,1,1]).astype('float32')
            all_mask.append(mask)
        all_mask = torch.Tensor(np.concatenate(all_mask, 0))
        return all_mask
    
    def _generate_mask_by_classfication(self, hr, sr, mode, Dvan):
        assert mode in ["spr", "hpr"]
        n = hr.size(0)
        temp_sr = sr.detach()
        temp_sr.requires_grad = True

        sr_result = Dvan(temp_sr)

        loss_func = torch.mean(torch.abs(sr_result))

        loss_func.backward()

        patch_contribute = torch.mean(self.patch_mean(temp_sr.grad), dim=1, keepdim=True)

        patch_contribute = torch.reshape(patch_contribute, [n, -1])

        if isinstance(self.portion, int):
            value, index = torch.sort(patch_contribute, descending=True, dim=-1)
            if mode == "spr":
                threshold = float(value[int((1-self.portion)*len(index))])
                mask = torch.where(patch_contribute<=threshold, torch.ones_like(patch_contribute), torch.zeros_like(patch_contribute))
            elif mode == "hpr":
                threshold = float(value[int((1-self.portion)*len(index))])
                mask = torch.where(patch_contribute>=threshold, torch.ones_like(patch_contribute), torch.zeros_like(patch_contribute))
        else:
            if mode == "spr":
                mask = torch.where(patch_contribute<=0, torch.ones_like(patch_contribute), torch.zeros_like(patch_contribute))
            elif mode == "hpr":
                mask = torch.where(patch_contribute>=0, torch.ones_like(patch_contribute), torch.zeros_like(patch_contribute))
            mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()

        temp_sr.grad = None
        del loss_func

        return mask      

    def forward(self, HR, SR, real_portion=0.5, mode="random", Dvan=None, need_mask=False):

        self.portion =real_portion

        if random.uniform(0,1) < self.roll_prob:
            h_shift = int(self.r_size[0]/2)
            w_shift = int(self.r_size[1]/2)
            temp_hr = HR[:,:,h_shift:-h_shift,w_shift:-w_shift].detach()
            temp_sr = SR[:,:,h_shift:-h_shift,w_shift:-w_shift].detach()
            self.isroll = True
        else:
            temp_hr = HR.detach()
            temp_sr = SR.detach()
            self.isroll = False

        sr_patches = self._get_patches(temp_sr)

        if mode == "random":
            mask = self._generat_mask_random(sr_patches.size()[0], sr_patches.size()[1])
            mask = mask.cuda()
            mask = torch.ones_like(sr_patches) * mask
            mask = self._restore_image(mask)
        elif "spr" in mode or "hpr" in mode:
            mask = self._generate_mask_by_classfication(temp_hr, temp_sr, mode, Dvan.eval())
            mask = mask.cuda()
            mask = torch.ones_like(sr_patches) * mask
            mask = self._restore_image(mask)

        if self.isroll:
            mask = F.pad(mask, (w_shift,w_shift,h_shift,h_shift), mode='replicate')

        rp_image = (mask * HR) + ((torch.ones_like(mask) - mask) * SR)

        if need_mask:
            del temp_hr, temp_sr
            return rp_image, mask
        else:
            del temp_hr, temp_sr, mask
            return rp_image
        

class Replacement_v2(nn.Module):
    def __init__(self, r_size=(16,16), size=(128,128), roll_prob=0.5):
        super(Replacement_v2, self).__init__()
        self.r_size = r_size
        self.roll_prob = roll_prob

        self.patch_mean = nn.AvgPool2d(kernel_size=r_size, stride=r_size, padding=0)
        self.unfold = nn.Unfold(r_size, 1, 0, r_size)

        self.fold = nn.Fold(output_size=size, kernel_size=r_size, stride=r_size)
        self.fold_roll = nn.Fold(output_size=(int(size[0]-r_size[0]), int(size[1]-r_size[1])), kernel_size=r_size, stride=r_size)

        self.D_LOSS = loss.D_LOSS()

    def _get_patches(self, input):
        n,c,h,w = input.size()
        current = self.unfold(input)
        current = current.permute([0,2,1])
        current = torch.reshape(current,[n,-1,c,self.r_size[0],self.r_size[1]])
        return current

    def _restore_image(self, input):
        n, num_patch, _, _, _ = input.size()
        current = torch.reshape(input, [n, num_patch, -1])
        current = current.permute(0,2,1)
        if not self.isroll:
            current = self.fold(current)
        else:
            current = self.fold_roll(current)
        return current

    def _generat_mask_random(self, n, length):
        all_mask = []
        r_num = int(self.portion * length)
        for i in range(n):
            mask = np.ones(length)
            inplace = random.sample(range(0, length), r_num)
            mask = np.multiply(mask, list(i in inplace for i in range(length)))
            mask = np.reshape(mask,[1,length,1,1,1]).astype('float32')
            all_mask.append(mask)
        all_mask = torch.Tensor(np.concatenate(all_mask, 0))
        return all_mask
    
    def _get_classfication_contribute(self, input, Dvan):
        temp = input.detach()
        temp.requires_grad = True

        cls = Dvan(temp)

        loss = torch.mean(torch.abs(cls))
        loss.backward()

        return temp.grad.detach()
    
    def _generate_mask_by_classfication(self, hr, sr, mode, Dvan):
        assert mode in ["spr", "hpr", "spr-a"]
        n = hr.size(0)
        
        sr_cls_score = self._get_classfication_contribute(sr, Dvan)
        hr_cls_score = self._get_classfication_contribute(hr, Dvan)

        if self.r_size == (1,1):
            if mode == "hpr":
                mask = torch.where(sr_cls_score >= 0, torch.ones_like(sr_cls_score), torch.zeros_like(sr_cls_score))
            elif mode == "spr" or\
                 mode == "spr-a":
                mask = torch.where(sr_cls_score <= 0, torch.ones_like(sr_cls_score), torch.zeros_like(sr_cls_score))

            return mask
        
        sr_patch_contribute = torch.mean(self.patch_mean(sr_cls_score), dim=1, keepdim=True)
        hr_patch_contribute = torch.mean(self.patch_mean(hr_cls_score), dim=1, keepdim=True)

        sr_patch_contribute = torch.reshape(sr_patch_contribute, [n, -1])
        hr_patch_contribute = torch.reshape(hr_patch_contribute, [n, -1])

        if isinstance(self.portion, int):
            value, index = torch.sort(sr_patch_contribute, descending=True, dim=-1)
            if mode == "spr":
                threshold = float(value[int((1-self.portion)*len(index))])
                mask = torch.where(sr_patch_contribute<=threshold, torch.ones_like(sr_patch_contribute), torch.zeros_like(sr_patch_contribute))
            elif mode == "hpr":
                threshold = float(value[int((1-self.portion)*len(index))])
                mask = torch.where(sr_patch_contribute>=threshold, torch.ones_like(sr_patch_contribute), torch.zeros_like(sr_patch_contribute))
        else:
            if mode == "spr":
                mask = torch.where(sr_patch_contribute<=0, torch.ones_like(sr_patch_contribute), torch.zeros_like(sr_patch_contribute))
            elif mode == "hpr":
                mask = torch.where(sr_patch_contribute>=0, torch.ones_like(sr_patch_contribute), torch.zeros_like(sr_patch_contribute))
            elif mode == "spr-a":
                hr_patch_contribute = hr_patch_contribute / torch.max(torch.abs(hr_patch_contribute), dim=-1, keepdim=True)[0]
                sr_patch_contribute = sr_patch_contribute / torch.max(torch.abs(sr_patch_contribute), dim=-1, keepdim=True)[0]
                synthetic_contribute = -hr_patch_contribute * 0.5 + sr_patch_contribute
                #value, index = torch.sort(synthetic_contribute, escending=True, dim=-1)
                mask = torch.where(synthetic_contribute<0, torch.ones_like(synthetic_contribute), torch.zeros_like(synthetic_contribute))


            mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()


        return mask      

    def _generate_checkboard_mask(self, hr: torch.Tensor) -> torch.Tensor:
        """
        Generate a checkerboard mask in r_size blocks.
        - hr: (B,C,H,W) or (C,H,W) or (H,W)
        - returns: mask shaped (B,1,H,W) (or broadcastable to hr), values in {0,1}
        """
        # Normalize to (B,C,H,W)
        if hr.dim() == 2:
            hr = hr.unsqueeze(0).unsqueeze(0)
        elif hr.dim() == 3:
            hr = hr.unsqueeze(0)
        elif hr.dim() != 4:
            raise ValueError(f"hr must be 2D/3D/4D tensor, got shape={tuple(hr.shape)}")

        B, _, H, W = hr.shape
        rh, rw = self.r_size

        # We assume H,W match the class 'size' used to build self.fold;
        # but we can still safely handle H,W that are multiples of r_size.
        if (H % rh) != 0 or (W % rw) != 0:
            raise ValueError(f"H,W must be multiples of r_size. Got (H,W)=({H},{W}), r_size=({rh},{rw})")

        gh, gw = H // rh, W // rw  # grid size in blocks

        # Checkerboard on the block grid: (gh, gw) with 0/1 alternating
        device = hr.device
        dtype = hr.dtype
        yy = torch.arange(gh, device=device)
        xx = torch.arange(gw, device=device)
        grid = (yy[:, None] + xx[None, :]) % 2  # 0/1 checkerboard
        grid = grid.to(dtype=dtype)  # float mask

        # Expand each grid cell to an rh x rw block
        # -> (H,W)
        mask2d = grid.repeat_interleave(rh, dim=0).repeat_interleave(rw, dim=1)

        # Optional "roll" by half a block to randomize phase (matches your fold_roll idea)
        # half-block shift is common; you can change shift sizes if you want.
        if self.roll_prob > 0:
            do_roll = (torch.rand(1, device=device).item() < float(self.roll_prob))
            if do_roll:
                shift_y = rh // 2
                shift_x = rw // 2
                if shift_y > 0 or shift_x > 0:
                    mask2d = torch.roll(mask2d, shifts=(shift_y, shift_x), dims=(0, 1))

        # Return (B,1,H,W) broadcastable to hr
        mask = mask2d.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W).contiguous()
        return mask

    def forward(self, HR, SR, real_portion=0.5, mode="random", Dvan=None, need_mask=False, need_reserve_image=False):

        self.portion =real_portion

        if random.uniform(0,1) < self.roll_prob:
            h_shift = int(self.r_size[0]/2)
            w_shift = int(self.r_size[1]/2)
            temp_hr = HR[:,:,h_shift:-h_shift,w_shift:-w_shift].detach()
            temp_sr = SR[:,:,h_shift:-h_shift,w_shift:-w_shift].detach()
            self.isroll = True
        else:
            temp_hr = HR.detach()
            temp_sr = SR.detach()
            self.isroll = False

        sr_patches = self._get_patches(temp_sr)

        if mode == "random":
            mask = self._generat_mask_random(sr_patches.size()[0], sr_patches.size()[1])
            mask = mask.cuda()
            mask = torch.ones_like(sr_patches) * mask
            mask = self._restore_image(mask)
        elif "spr" in mode or "hpr" in mode:
            mask = self._generate_mask_by_classfication(temp_hr, temp_sr, mode, Dvan.eval())
            mask = mask.cuda()
            mask = torch.ones_like(sr_patches) * mask
            mask = self._restore_image(mask)
        elif mode == 'checkboard':
            mask = self._generate_checkboard_mask(HR)

            if random.uniform(0, 1) < 0.5:
                mask = torch.ones_like(mask) - mask
            
            mask = mask.cuda()

        if self.isroll:
            mask = F.pad(mask, (w_shift,w_shift,h_shift,h_shift), mode='replicate')

        rp_image = (mask * HR) + ((torch.ones_like(mask) - mask) * SR)
        rp_image_reverse = ((torch.ones_like(mask) - mask) * HR) + (mask * SR)

        res = [rp_image]

        if need_mask:
            res.append(mask)

        if need_reserve_image:
            res.append(rp_image_reverse)

        if len(res) == 1:
            return res[0]
        return res

