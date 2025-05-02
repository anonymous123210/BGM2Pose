import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .wisppn_resnet import get_wisppn
import torch
from .functions import get_downsampling_block1, get_downsampling_block2_1d, get_downsampling_block1_1d, get_downsampling_block2
import numpy as np

class Speech2pose_freq_atten_ver10(nn.Module):
    def __init__(self, in_cha = 4, out_cha = 4, dropout =0.1):
        super().__init__()
        self.cha_1 = 64
        self.down_k = 4
        self.k = 3
        self.down_s = 2
        self.s = 1
        self.cha_2 = 128
        self.cha_3 = 256
        self.cha_4 = 256
        self.new_freq_dim = 16
        self.main_block = []
        self.main_block.append(get_downsampling_block1(in_cha, self.cha_1, k = self.k, s = 1, p = None))
        self.main_block.append(get_downsampling_block1(self.cha_1, self.cha_2, k = self.k, s = 1, p = None))
        self.main_block = nn.Sequential(*self.main_block)
        
        self.music_block = []
        self.music_block.append(get_downsampling_block1(2, self.cha_1, k = self.k, s = 1, p = None))
        self.music_block.append(get_downsampling_block1(self.cha_1, self.cha_2, k = self.k, s = 1, p = None))
        self.music_block = nn.Sequential(*self.music_block)
        
        self.multi_head_attention1 = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.1)
        self.multi_head_attention2 = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        
        self.main_block2 = []
        self.main_block2.append(get_downsampling_block1(self.cha_2, self.cha_3, k = self.k, s = 1, p = (1, 2)))
        self.main_block2.append(get_downsampling_block1(self.cha_3, self.cha_4, k = self.k, s = 1, p=(1,2)))
        self.main_block2.append(get_downsampling_block1(self.cha_4, self.cha_4, k = self.k, s = 1, p=(1, 2)))
        self.main_block2 = nn.Sequential(*self.main_block2)
        
        self.fifth_block = get_downsampling_block2_1d(self.cha_4 * self.new_freq_dim, self.cha_4, k = 2, s = self.s, p=None)
        self.sixth_block = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = 2, s = self.s)
        self.seventh_block = get_downsampling_block1_1d(self.cha_4, self.cha_4, k = 2, s = self.s)
        self.eight_block = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s)
        self.ninth_block = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s)
        self.tenth_block = get_downsampling_block2(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.up1 = nn.ConvTranspose1d(in_channels=self.cha_4, out_channels=self.cha_4, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(in_channels=self.cha_4, out_channels=self.cha_4, kernel_size=4, stride=2, padding=1)
        self.dec1 = get_downsampling_block2_1d(self.cha_4 * 2, self.cha_4, k = self.k, s = self.s, p = None)
        self.dec2 = get_downsampling_block2_1d(self.cha_4 * 2, self.cha_4, k = self.k, s = self.s, p = None)
        self.fifth_block2 = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.fifth_block3 = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.fifth_block4 = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.fifth_block5 = get_downsampling_block2_1d(self.cha_4, self.cha_4, k = self.k, s = self.s, p = None)
        self.bottleneck = nn.Conv1d(self.cha_4, out_cha, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(128, self.cha_2)
        self.pos_embedding2 = nn.Embedding(32, self.cha_4)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cha_4))
        self.sound_seq_fusion = nn.MultiheadAttention(embed_dim=self.cha_4, num_heads=8, dropout=0.1)
        self.pose_embedding = get_downsampling_block2_1d(63, self.cha_4, k = 3, s = 1, p=None)
        self.pose_mixer = nn.MultiheadAttention(embed_dim=self.cha_4, num_heads=8, dropout=0.1)
        self.pose_token = nn.Parameter(torch.randn(1, 1, self.cha_4))
        self.time_embedding = nn.Embedding(13, self.cha_4)
        init_logit_scale = np.log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale, requires_grad=False)
        
    def forward(self, mel, poses): #mel -> (bs, in_cha = 9, time, freq)
        bs, in_cha, seq_len, freq_num = mel.shape
        # Add time embedding
        time_id = torch.arange(seq_len + 1, device=mel.device).unsqueeze(0)
        time_embedding = self.time_embedding(time_id.long()) #[1, 12, cha_4]
        time_embedding = time_embedding.permute(1, 0, 2)
        pose_tokens = self.pose_token.expand(bs, -1, -1)  # (B, 1, E)
        pose_tokens = pose_tokens.permute(1, 0, 2)
        poses = poses.permute(0, 2, 1) #[bs, 63, seq_len]
        poses = self.pose_embedding(poses) #[bs, 128, seq_len]
        poses = poses.permute(2, 0, 1)
        poses = torch.cat([pose_tokens, poses], dim=0)
        poses = poses + time_embedding
        poses, _ = self.pose_mixer(query=poses, key=poses, value=poses)
        poses = poses[0]
        
        mel = self.dropout(mel)
        diff1 = mel[:,:4,:,:] - mel[:,4,:,:].reshape(mel.shape[0], 1, mel.shape[-2], mel.shape[-1])
        diff2 = mel[:,:4,:,:] - mel[:,5,:,:].reshape(mel.shape[0], 1, mel.shape[-2], mel.shape[-1])
        music_feats = mel[:,4:6,:,:]
        mel = torch.cat([diff1, diff2, mel[:,-3:,:,:]], dim=1)
        out = self.main_block(mel)
        out = self.dropout(out) 
        out = out.permute(0, 2, 3, 1).reshape(-1, freq_num, self.cha_2)
        pos_id = torch.arange(freq_num).unsqueeze(0).to(mel).long()
        pos_x = self.pos_embedding(pos_id) #[1, freq_num, cha_2]
        out = out + pos_x
        out = out.permute(1,0,2) #[freq, bs', cha_2]
        
        music_feats = self.music_block(music_feats)
        music_feats = self.dropout(music_feats) 
        music_feats = music_feats.permute(0, 2, 3, 1).reshape(-1, freq_num, self.cha_2)
        pos_id = torch.arange(freq_num).unsqueeze(0).to(mel).long()
        
        pos_x_music = self.pos_embedding(pos_id) #[1, freq_num, cha_2]
        music_feats = music_feats + pos_x_music
        music_feats = music_feats.permute(1,0,2) #[freq, bs', cha_2]
        
        att_out, att_weight = self.multi_head_attention1(query=music_feats, key=out, value=out)
        att_out = att_out.permute(1,0,2) #[bs' freq, cha_2]
        att_out = att_out.reshape(bs, seq_len, freq_num, self.cha_2)
        att_out = att_out.permute(0, 3, 1, 2)
        out = att_out + out.permute(1,0,2).reshape(bs, seq_len, freq_num, self.cha_2).permute(0, 3, 1, 2)

        
        
        out = self.main_block2(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1, seq_len, 1) #[bs, seq_len, 1, channels * f']
        out_4 = out.squeeze(-1) #[bs, channel * freq, seq_len] #[256, 4096, 12]
        out_5 = self.fifth_block(out_4) #[bs, 256, 12]
        latents = out_5
        
        
        cls_tokens = self.cls_token.expand(bs, -1, -1)  # (B, 1, E)
        cls_tokens = cls_tokens.permute(1, 0, 2)
        
        out_5_ = torch.cat([cls_tokens, out_5.permute(2, 0, 1)], dim=0)
        
        out_5_ = out_5_ + time_embedding
        out_5_, _ = self.sound_seq_fusion(query=out_5_, key=out_5_, value=out_5_)
        seq_emb = out_5_[0] #[bs, cha_4]
        
        out_6 = self.sixth_block(out_5) #[256, 256, 6]
        out_6 = self.dropout(out_6)
        out_7 = self.seventh_block(out_6) #[256, 256, 3]
        out_6 = torch.concat([out_6, self.up1(out_7)], dim=1) #[256, 512, 6]
        out_6 = self.dec1(out_6)
        out_5 = torch.concat([out_5, self.up2(out_6)], dim=1)
        out_5 = self.dec2(out_5)
        out_5 = self.fifth_block2(out_5)
        out_5 = self.fifth_block3(out_5)
        out_5 = self.dropout(out_5)
        out_5 = self.fifth_block4(out_5)
        out_5 = self.dropout(out_5)
        out_5 = self.fifth_block5(out_5) #[bs,seq_len, channels]
        output = self.bottleneck(out_5) #[bs, 21 * 3, seq_len]
        output = output.permute(0, 2, 1) #[bs, seq_len, 21 * 3]
        return output, F.normalize(seq_emb, dim=-1), F.normalize(poses, dim=-1), self.logit_scale.exp(), latents