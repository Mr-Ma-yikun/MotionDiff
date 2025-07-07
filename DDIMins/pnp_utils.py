#import ot
import torch
import os
import random
import numpy as np
import torchvision
from diffusers.models.attention import _chunked_feed_forward
from matplotlib import pyplot as plt


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0]
            # module2 = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
            # setattr(module2, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            module2 = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
            setattr(module2, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    module2 = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)
    setattr(module2, 't', t)


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents

def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        attn1_to_out = self.attn1.to_out
        attn2_to_out = self.attn2.to_out
        if type(attn1_to_out) is torch.nn.modules.container.ModuleList:
            attn1_to_out = self.attn1.to_out[0]
        else:
            attn1_to_out = self.attn1.to_out
        if type(attn2_to_out) is torch.nn.modules.container.ModuleList:
            attn2_to_out = self.attn2.to_out[0]
        else:
            attn2_to_out = self.attn2.to_out

        def forward(x, timestep=None,
                    cross_attention_kwargs=None,
                    encoder_attention_mask=None,
                    encoder_hidden_states=None,
                    attention_mask=None,
                    class_labels=None):
            batch_size, sequence_length, dim = x.shape
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(x, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    x, timestep, class_labels, hidden_dtype=x.dtype
                )
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(x)
            elif self.use_ada_layer_norm_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(x)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")
            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            # Retrieve lora scale.
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
            # Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            # self-attention-----------------------
            attn1_h = self.attn1.heads
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                q = self.attn1.to_q(norm_hidden_states)
                k = self.attn1.to_k(norm_hidden_states)

                source_batch_size = int(q.shape[0] // 4)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:3 * source_batch_size] = q[:source_batch_size]
                k[2 * source_batch_size:3 * source_batch_size] = k[:source_batch_size]

                q = self.attn1.head_to_batch_dim(q)
                k = self.attn1.head_to_batch_dim(k)
            else:
                q = self.attn1.to_q(norm_hidden_states)
                k = self.attn1.to_k(norm_hidden_states)
                q = self.attn1.head_to_batch_dim(q)
                k = self.attn1.head_to_batch_dim(k)

            v = self.attn1.to_v(norm_hidden_states)
            # if self.t > 800:
            #     v[4:5] = v[2:3]
            #     v[1:2] = v[3:4]
            #     # q[2:3] = q[4:5]
            #     # q[1:2] = q[3:4]
            #     k[2:3] = k[4:5]
            #     k[1:2] = k[3:4]
            v = self.attn1.head_to_batch_dim(v)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.attn1.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(attn1_h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.attn1.batch_to_head_dim(out)
            attn1_out = attn1_to_out(out)
            #-------------------------------------

            if self.use_ada_layer_norm_zero:
                attn1_out = gate_msa.unsqueeze(1) * attn1_out
            elif self.use_ada_layer_norm_single:
                attn1_out = gate_msa * attn1_out
            hidden_states = attn1_out + x
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
            # GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # cross attention-------------------
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")
            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            batch_size, sequence_length, dim = norm_hidden_states.shape
            attn2_h = self.attn2.heads
            q = self.attn2.to_q(norm_hidden_states)
            k = self.attn2.to_k(encoder_hidden_states)
            v = self.attn2.to_v(encoder_hidden_states)


            # source_batch_size = int(q.shape[0] // 4)
            # # inject unconditional
            # q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
            # k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
            # v[source_batch_size:2 * source_batch_size] = v[:source_batch_size]
            # # inject conditional
            # q[2 * source_batch_size:3 * source_batch_size] = q[:source_batch_size]
            # k[2 * source_batch_size:3 * source_batch_size] = k[:source_batch_size]
            # v[2 * source_batch_size:3 * source_batch_size] = v[:source_batch_size]

            # OT-----
            # n, s, c = k.shape
            # X1 = k[2:3].reshape(-1, c).detach().cpu().numpy()
            # X2 = k[5:6].reshape(-1, c).detach().cpu().numpy()
            # nb = min(len(X1) // 2, 500)
            # rng = np.random.RandomState(42)
            # idx1 = rng.randint(X1.shape[0], size=(nb,))
            # idx2 = rng.randint(X2.shape[0], size=(nb,))
            # Xs = X1[idx1, :]
            # Xt = X2[idx2, :]
            # # EMDTransport
            # ot_emd = ot.da.EMDTransport()
            # ot_emd.fit(Xs=Xs, Xt=Xt)
            # transp_Xs_emd = ot_emd.transform(Xs=X1)
            # Image_emd = transp_Xs_emd.reshape(s, c)
            # self.alpha = 0.0
            # # v[2:3]+=100.0
            # k[2:3] = self.alpha * k[2:3] + (1 - self.alpha) * torch.tensor(Image_emd).cuda().unsqueeze(0)
            # # OT-----

            q = self.attn2.head_to_batch_dim(q)
            k = self.attn2.head_to_batch_dim(k)
            v = self.attn2.head_to_batch_dim(v)


            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.attn2.scale


            # # environment mask
            # attn = sim.softmax(dim=-1)
            # # # import torch.nn.functional as F
            # vis_att = sim[:, :, 1].softmax(dim=-1)
            # ratio = int(np.sqrt(5440.0 // sequence_length + 1))
            # # # print(ratio)
            # num = attn.shape[0] // 4
            # attention_mask = torch.mean(vis_att.reshape(-1,batch_size, int(np.ceil(85 / ratio)),
            #                                                           64 // ratio), dim=0, keepdim=False)
            # attention_mask = torch.gt(attention_mask, torch.mean(attention_mask, dim=[1,2], keepdim=True))

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(attn2_h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            # # import torch.nn.functional as F
            vis_att = sim[:, :, 4].softmax(dim=-1)
            ratio = int(np.sqrt(4096.0 // sequence_length + 1))
            # # print(ratio)
            num = attn.shape[0] // 4
            show_att = torch.mean(vis_att[num * 2:num * 3, :].reshape(num, int(np.ceil(64 / ratio)),
                                                                      64 // ratio).unsqueeze(1), dim=0,
                                  keepdim=True)
            # # show_att = (self.atten - torch.min(self.atten)) / (torch.max(self.atten) - torch.min(self.atten))
            # plt.imshow(show_att[0][0].detach().cpu().numpy(), cmap='jet')
            # plt.savefig(f'E:\\low_level_package\\pnp-diffusers-main_copy\\PNP-results\\cat\\temp2\\{ratio}\\{self.t}.png')
            torchvision.utils.save_image(show_att * 255,
                                         f'E:\\low_level_package\\pnp-diffusers-main_copy\\PNP-results\\cat\\temp2\\{ratio}.png')

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.attn2.batch_to_head_dim(out)
            attn2_output = attn2_to_out(out)
            hidden_states = attn2_output + hidden_states
            # FF----------------
            # 4. Feed-forward
            if not self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(
                    self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
                )
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.use_ada_layer_norm_single:
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
            return hidden_states
        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0]
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_cross_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 4)
                q_source = q[:source_batch_size]
                k_source = k[:source_batch_size]
                # q_target = q[4 * source_batch_size:]
                # k_target = k[4 * source_batch_size:]
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q_source + 0.1
                k[source_batch_size:2 * source_batch_size] = k_source + 0.1
                # inject conditional
                q[2 * source_batch_size:3 * source_batch_size] = q_source + 0.1
                k[2 * source_batch_size:3 * source_batch_size] = k_source + 0.1
                # inject target
                # q[3 * source_batch_size:4 * source_batch_size] = q_target + 0.1
                # k[3 * source_batch_size:4 * source_batch_size] = k_target + 0.1

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            if is_cross:
                import torch.nn.functional as F
                vis_att = sim[:,:,8].softmax(dim=-1)
                ratio = int(np.sqrt(5440.0 // sequence_length + 1))
                # print(ratio)
                num = attn.shape[0] // 4
                show_att = torch.mean(vis_att[num*3:num*4, :].reshape(num, int(np.ceil(85 / ratio)),
                                                                   64 // ratio).unsqueeze(1), dim=0, keepdim=True)
                # show_att = (self.atten - torch.min(self.atten)) / (torch.max(self.atten) - torch.min(self.atten))
                # plt.imshow(show_att[0][0].detach().cpu().numpy(), cmap='jet')
                # plt.savefig(f'E:\\low_level_package\\pnp-diffusers-main_copy\\PNP-results\\cat\\temp2\\{ratio}\\{self.t}.png')
                torchvision.utils.save_image(show_att*255,
                                             f'E:\\low_level_package\\pnp-diffusers-main_copy\\PNP-results\\cat\\temp2\\{ratio}.png')
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # res_dict = {1: [1, 2], 2: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb, scale=1.0):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 4)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:3 * source_batch_size] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)