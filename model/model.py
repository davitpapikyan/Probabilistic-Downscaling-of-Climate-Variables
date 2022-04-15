"""Denoising Diffusion Probabilistic Model.

Combines U-Net network with Denoising Diffusion Model and
creates single image super-resolution solver architecture.

The work is based on https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.
"""
import logging
import os
import typing
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from .base_model import BaseModel
from .ema import EMA
from .networks import define_network

logger = logging.getLogger("base")


class DDPM(BaseModel):
    """Denoising Diffusion Probabilistic Model.

    Attributes:
        in_channel: The number of channels of input tensor of U-Net.
        out_channel: The number of channels of output tensor of U-Net.
        norm_groups: The number of groups for group normalization.
        inner_channel: Timestep embedding dimension.
        channel_multiplier: A tuple specifying the scaling factors of channels.
        attn_res: A tuple of spatial dimensions indicating in which resolutions to use self-attention layer.
        res_blocks: The number of residual blocks.
        dropout: Dropout probability.
        diffusion_loss: Either l1 or l2.
        conditional: Whether to condition on INTERPOLATED image or not.
        gpu_ids: IDs of gpus.
        distributed: Whether the computation will be distributed among multiple GPUs or not.
        init_method: NN weight initialization method. One of normal, kaiming or orthogonal inisializations.
        train_schedule: Defines the type of beta schedule for training.
        train_n_timestep: Number of diffusion timesteps for training.
        train_linear_start: Minimum value of the linear schedule for training.
        train_linear_end: Maximum value of the linear schedule for training.
        val_schedule: Defines the type of beta schedule for validation.
        val_n_timestep: Number of diffusion timesteps for validation.
        val_linear_start: Minimum value of the linear schedule for validation.
        val_linear_end: Maximum value of the linear schedule for validation.
        finetune_norm: Whetehr to fine-tune or train from scratch.
        optimizer: The optimization algorithm.
        amsgrad: Whether to use the AMSGrad variant of optimizer.
        learning_rate: The learning rate.
        checkpoint: Path to the checkpoint file.
        resume_state: The path to load the network.
        phase: Either train or val.
        height: U-Net input tensor height value.
    """

    def __init__(self, in_channel, out_channel, norm_groups, inner_channel,
                 channel_multiplier, attn_res, res_blocks, dropout,
                 diffusion_loss, conditional, gpu_ids, distributed, init_method,
                 train_schedule, train_n_timestep, train_linear_start, train_linear_end,
                 val_schedule, val_n_timestep, val_linear_start, val_linear_end,
                 finetune_norm, optimizer, amsgrad, learning_rate, checkpoint, resume_state,
                 phase, height):

        super(DDPM, self).__init__(gpu_ids)
        noise_predictor = define_network(in_channel, out_channel, norm_groups, inner_channel,
                                         channel_multiplier, attn_res, res_blocks, dropout,
                                         diffusion_loss, conditional, gpu_ids, distributed,
                                         init_method, height)
        self.SR_net = self.set_device(noise_predictor)
        self.loss_type = diffusion_loss
        self.data, self.SR = None, None
        self.checkpoint = checkpoint
        self.resume_state = resume_state
        self.finetune_norm = finetune_norm
        self.phase = phase
        self.set_loss()
        self.months = []  # A list of months of curent data given by feed_data.

        if self.phase == "train":
            self.set_new_noise_schedule(schedule=train_schedule, n_timestep=train_n_timestep,
                                        linear_start=train_linear_start, linear_end=train_linear_end)
        else:
            self.set_new_noise_schedule(schedule=val_schedule, n_timestep=val_n_timestep,
                                        linear_start=val_linear_start, linear_end=val_linear_end)

        if self.phase == "train":
            self.SR_net.train()
            if self.finetune_norm:
                optim_params = []
                for k, v in self.SR_net.named_parameters():
                    v.requires_grad = False
                    if k.find("transformer") >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(f"Params [{k:s}] initialized to 0 and will be fine-tuned.")
            else:
                optim_params = list(self.SR_net.parameters())

            self.optimizer = optimizer(optim_params, lr=learning_rate, amsgrad=amsgrad)

            # Learning rate schedulers.
            # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=40000, eta_min=1e-6)
            self.scheduler = MultiStepLR(self.optimizer, milestones=[40000], gamma=0.5)

            self.ema = EMA(mu=0.9999)
            self.ema.register(self.SR_net)

            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

    def feed_data(self, data: tuple) -> None:
        """Stores data for feeding into the model and month indices for each tensor in a batch.

        Args:
            data: A tuple containing dictionary with the following keys:
                HR: a batch of high-resolution images [B, C, H, W],
                LR: a batch of low-resolution images [B, C, H, W],
                INTERPOLATED: a batch of upsampled (via interpolation) images [B, C, H, W]
            and list of corresponding months of samples in a batch.
        """
        self.data, self.months = self.set_device(data[0]), data[1]

    def optimize_parameters(self) -> None:
        """Computes loss and performs GD step on learnable parameters.
        """
        self.optimizer.zero_grad()
        loss = self.SR_net(self.data)
        loss = loss.sum() / self.data["HR"].numel()
        loss.backward()
        self.optimizer.step()
        self.ema.update(self.SR_net)  # Exponential Moving Average step of parameters.
        self.log_dict[self.loss_type] = loss.item()  # Setting the log.

    def lr_scheduler_step(self):
        """Learning rate scheduler step.
        """
        # self.scheduler.step()

    def get_lr(self) -> float:
        """Fetches current learning rate.

        Returns:
            Current learning rate value.
        """
        return self.optimizer.param_groups[0]['lr']

    def get_named_parameters(self) -> dict:
        """Fetched U-Net's parameters.

        Returns:
            U-Net's parameters with their names.
        """
        return self.SR_net.named_parameters()

    def test(self, continuous: bool = False) -> None:
        """Constructs the super-resolution image and assiggns to SR attribute.

        Args:
            continuous: Either to return all the SR images for each denoising timestep or not.
        """
        self.SR_net.eval()
        with torch.no_grad():
            if isinstance(self.SR_net, nn.DataParallel):
                self.SR = self.SR_net.module.super_resolution(self.data["INTERPOLATED"], continuous)
            else:
                self.SR = self.SR_net.super_resolution(self.data["INTERPOLATED"], continuous)
            self.SR = self.SR.unsqueeze(0) if len(self.SR.size()) == 3 else self.SR

        self.SR_net.train()

    def generate_multiple_candidates(self, n: int = 10) -> torch.tensor:
        """Generates n super-resolution tesnors.

        Args:
            n: The number of candidates.

        Returns:
            n super-resolution tensors of shape [n, B, C, H, W] corresponding
            to data fed into the model.
        """
        self.SR_net.eval()
        batch_size, c, h, w = self.data["INTERPOLATED"].size()
        sr_candidates = torch.empty(size=(n, batch_size, c, h, w))
        with torch.no_grad():
            for i in range(n):
                if isinstance(self.SR_net, nn.DataParallel):
                    x_sr = self.SR_net.module.super_resolution(self.data["INTERPOLATED"], False).detach().float().cpu()
                else:
                    x_sr = self.SR_net.super_resolution(self.data["INTERPOLATED"], False).detach().float().cpu()
                sr_candidates[i] = x_sr.unsqueeze(0) if len(x_sr.size()) == 3 else x_sr

        self.SR_net.train()
        return sr_candidates

    def set_loss(self) -> None:
        """Sets loss to a device.
        """
        if isinstance(self.SR_net, nn.DataParallel):
            self.SR_net.module.set_loss(self.device)
        else:
            self.SR_net.set_loss(self.device)

    def set_new_noise_schedule(self, schedule, n_timestep, linear_start, linear_end) -> None:
        """Creates new noise scheduler.

        Args:
            schedule: Defines the type of beta schedule.
            n_timestep: Number of diffusion timesteps.
            linear_start: Minimum value of the linear schedule.
            linear_end: Maximum value of the linear schedule.
        """
        if isinstance(self.SR_net, nn.DataParallel):
            self.SR_net.module.set_new_noise_schedule(schedule, n_timestep, linear_start, linear_end, self.device)
        else:
            self.SR_net.set_new_noise_schedule(schedule, n_timestep, linear_start, linear_end, self.device)

    def get_current_log(self) -> OrderedDict:
        """Returns the logs.

        Returns:
            log_dict: Current logs of the model.
        """
        return self.log_dict

    def get_months(self) -> list:
        """Returns the list of month indices corresponding to batch of samples
        fed into the model with feed_data.

        Returns:
            months: Current list of months.
        """
        return self.months

    def get_current_visuals(self, need_LR: bool = True, only_rec: bool = False) -> typing.OrderedDict:
        """Returns only reconstructed super-resolution image if only_rec is True (with "SAM" key),
        otherwise returns super-resolution image (with "SR" key), interpolated LR image
        (with "interpolated" key), HR image (with "HR" key), LR image (with "LR" key).

        Args:
            need_LR: Whether to return LR image or not.
            only_rec: Whether to return only reconstructed super-resolution image or not.

        Returns:
            Dict containing desired images.
        """
        out_dict = OrderedDict()
        if only_rec:
            out_dict["SR"] = self.SR.detach().float().cpu()
        else:
            out_dict["SR"] = self.SR.detach().float().cpu()
            out_dict["INTERPOLATED"] = self.data["INTERPOLATED"].detach().float().cpu()
            out_dict["HR"] = self.data["HR"].detach().float().cpu()
            if need_LR and "LR" in self.data:
                out_dict["LR"] = self.data["LR"].detach().float().cpu()
        return out_dict

    def print_network(self) -> None:
        """Prints the network architecture.
        """
        s, n = self.get_network_description(self.SR_net)
        if isinstance(self.SR_net, nn.DataParallel):
            net_struc_str = "{} - {}".format(self.SR_net.__class__.__name__, self.SR_net.module.__class__.__name__)
        else:
            net_struc_str = "{}".format(self.SR_net.__class__.__name__)

        logger.info(f"U-Net structure: {net_struc_str}, with parameters: {n:,d}")
        logger.info(f"Architecture:\n{s}\n")

    def save_network(self, epoch: int, iter_step: int) -> None:
        """Saves the network checkpoint.

        Args:
            epoch: How many epochs has the model been trained.
            iter_step: How many iteration steps has the model been trained.
        """
        gen_path = os.path.join(self.checkpoint, f"I{iter_step}_E{epoch}_gen.pth")
        opt_path = os.path.join(self.checkpoint, f"I{iter_step}_E{epoch}_opt.pth")

        network = self.SR_net.module if isinstance(self.SR_net, nn.DataParallel) else self.SR_net

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)

        opt_state = {"epoch": epoch, "iter": iter_step,
                     "scheduler": self.scheduler.state_dict(),
                     "optimizer": self.optimizer.state_dict()}
        torch.save(opt_state, opt_path)
        logger.info("Saved model in [{:s}] ...".format(gen_path))

    def load_network(self) -> None:
        """Loads the netowrk parameters.
        """
        if self.resume_state is not None:
            logger.info(f"Loading pretrained model for G [{self.resume_state:s}] ...")
            gen_path, opt_path = f"{self.resume_state}_gen.pth", f"{self.resume_state}_opt.pth"

            network = self.SR_net.module if isinstance(self.SR_net, nn.DataParallel) else self.SR_net
            network.load_state_dict(torch.load(gen_path), strict=(not self.finetune_norm))

            if self.phase == "train":
                opt = torch.load(opt_path)
                self.optimizer.load_state_dict(opt["optimizer"])
                self.scheduler.load_state_dict(opt["scheduler"])
                self.begin_step = opt["iter"]
                self.begin_epoch = opt["epoch"]
