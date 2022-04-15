"""The inference script for DDPM model for a single image.

This script is not a part of the pipeline of the project. It
was used to generate plots and statistics for a single
data sample case.
"""
import argparse
import logging
import os
import warnings
from collections import OrderedDict

import numpy as np
import torch
from torch.nn.functional import mse_loss, l1_loss
from torch.utils.data import DataLoader

import model
from config import Config, get_current_datetime
from utils import dict2str, setup_logger, construct_and_save_wbd_plots, \
    construct_mask, reverse_transform_candidates, set_seeds
from weatherbench_data import collate_wb_batch
from weatherbench_data.utils import reverse_transform, reverse_transform_tensor, load_object, prepare_test_data

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    set_seeds()  # For reproducability.

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    parser.add_argument("-p", "--phase", type=str, choices=["train", "val"],
                        help="Run either training or validation(inference).", default="train")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    args = parser.parse_args()
    configs = Config(args)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    test_root = f"{configs.experiments_root}/test_on_single_image_{get_current_datetime()}"
    os.makedirs(test_root, exist_ok=True)
    setup_logger("test", test_root, "test", screen=True)
    val_logger = logging.getLogger("test")
    val_logger.info(dict2str(configs.get_hyperparameters_as_dict()))

    # Preparing testing data.
    transformations = load_object(configs.experiments_root, "transformations")
    metadata = load_object(configs.experiments_root, "metadata")
    if transformations and metadata:
        val_logger.info("Transformations and Metadata are successfuly loaded.")

    val_dataset = prepare_test_data(variables=configs.variables, val_min_date=configs.val_min_date,
                                    val_max_date=configs.val_max_date, dataroot=configs.dataroot,
                                    transformations=transformations)

    val_logger.info(f"Dataset [{val_dataset.__class__.__name__} - Testing] is created.")
    val_logger.info(f"""Created {val_dataset.__class__.__name__} dataset of length {len(val_dataset)}, 
    containing data from {val_dataset.min_date} until {val_dataset.max_date}""")
    val_logger.info(f"Group structure: {val_dataset.get_data_names()}")
    val_logger.info(f"Channel count: {val_dataset.get_channel_count()}\n")

    # The batch size is 1.
    val_loader = DataLoader(val_dataset,
                            collate_fn=collate_wb_batch,
                            pin_memory=True, drop_last=True,
                            num_workers=configs.num_workers)
    val_logger.info("Testing dataset is ready.")

    # Defining the model.
    diffusion = model.create_model(in_channel=configs.in_channel, out_channel=configs.out_channel,
                                   norm_groups=configs.norm_groups, inner_channel=configs.inner_channel,
                                   channel_multiplier=configs.channel_multiplier, attn_res=configs.attn_res,
                                   res_blocks=configs.res_blocks, dropout=configs.dropout,
                                   diffusion_loss=configs.diffusion_loss, conditional=configs.conditional,
                                   gpu_ids=configs.gpu_ids, distributed=configs.distributed,
                                   init_method=configs.init_method, train_schedule=configs.train_schedule,
                                   train_n_timestep=configs.train_n_timestep,
                                   train_linear_start=configs.train_linear_start,
                                   train_linear_end=configs.train_linear_end,
                                   val_schedule=configs.val_schedule, val_n_timestep=configs.val_n_timestep,
                                   val_linear_start=configs.val_linear_start, val_linear_end=configs.val_linear_end,
                                   finetune_norm=configs.finetune_norm, optimizer=None, amsgrad=configs.amsgrad,
                                   learning_rate=configs.lr, checkpoint=configs.checkpoint,
                                   resume_state=configs.resume_state, phase=configs.phase, height=configs.height)
    val_logger.info("Model initialization is finished.")

    current_step, current_epoch = diffusion.begin_step, diffusion.begin_epoch
    val_logger.info(f"Testing the model at epoch: {current_epoch}, iter: {current_step}.")

    diffusion.set_new_noise_schedule(schedule=configs.test_schedule,
                                     n_timestep=configs.test_n_timestep,
                                     linear_start=configs.test_linear_start,
                                     linear_end=configs.test_linear_end)
    accumulated_statistics = OrderedDict()

    # Creating placeholder for storing validation metrics Mean Squared Error, Root MSE, Mean Residual.
    val_metrics = OrderedDict({"MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "MR": 0.0,
                               "mean_bias_over_pixels": 0.0, "std_bias_over_pixels": 0.0})
    idx = 0

    result_path = f"{test_root}/results"
    os.makedirs(result_path, exist_ok=True)

    with torch.no_grad():
        for val_data in val_loader:
            idx += 1

            # Works only for one image.
            if idx % configs.val_vis_freq == 0:
                diffusion.feed_data(val_data)
                val_logger.info("Starting to generate SR images.")
                diffusion.test(continuous=True)  # Continues=False to return only the last timesteps's outcome.
                val_logger.info("Finished generating SR images.")

                # Computing metrics on vlaidation data..
                visuals = diffusion.get_current_visuals()
                # When continuous is True, visuals["SR"] has [T, C, H, W] dimension
                # where T is the number of diffusion timesteps.

                inv_visuals = reverse_transform(visuals, transformations,
                                                configs.variables, diffusion.get_months(),
                                                configs.tranform_monthly)

                # Computing MSE and RMSE on original data.
                mse_value = mse_loss(inv_visuals["HR"], inv_visuals["SR"][-1])
                val_metrics["MSE"] += mse_value
                val_metrics["RMSE"] += torch.sqrt(mse_value)
                val_metrics["MAE"] += l1_loss(inv_visuals["HR"], inv_visuals["SR"][-1])

                # Computing residuals for visualization.
                residuals = inv_visuals["SR"][-1] - inv_visuals["HR"]
                val_metrics["MR"] += residuals.mean()

                path = f"{result_path}/{idx}/"
                os.makedirs(path, exist_ok=True)
                path = f"{path}{idx}"

                val_logger.info("Started generating multiple SR candidates.")
                sr_candidates = diffusion.generate_multiple_candidates(n=configs.sample_size)
                reverse_transform_candidates(sr_candidates, reverse_transform_tensor,
                                             transformations, configs.variables,
                                             "hr", diffusion.get_months(),
                                             configs.tranform_monthly)
                val_logger.info("Finished generating multiple SR candidates.")

                mean_candidate = sr_candidates.mean(dim=0)
                std_candidate = sr_candidates.std(dim=0)
                bias = mean_candidate - inv_visuals["HR"]
                mean_bias_over_pixels = bias.mean()
                std_bias_over_pixels = bias.std()
                val_metrics["mean_bias_over_pixels"] += mean_bias_over_pixels
                val_metrics["std_bias_over_pixels"] += std_bias_over_pixels

                # Computing min and max measures to set a fixed colorbar for all visualizations.
                vmin = min(inv_visuals["HR"].min(),
                           inv_visuals["LR"].min(),
                           inv_visuals["INTERPOLATED"].min(),
                           mean_candidate.min())
                vmax = max(inv_visuals["HR"].max(),
                           inv_visuals["LR"].max(),
                           inv_visuals["INTERPOLATED"].max(),
                           mean_candidate.max())
                vmin, vmax = np.floor(vmin), np.ceil(vmax)

                val_logger.info(f"[{idx // configs.val_vis_freq}] Visualizing and storing some examples.")
                # Choosing the first n_val_vis number of samples to visualize.
                construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                             data=inv_visuals["HR"],
                                             path=f"{path}_hr.png", vmin=vmin, vmax=vmax)
                construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                             data=inv_visuals["SR"],
                                             path=f"{path}_sr.png", vmin=vmin, vmax=vmax)
                construct_and_save_wbd_plots(latitude=metadata.lr_lat, longitude=metadata.lr_lon,
                                             data=inv_visuals["LR"],
                                             path=f"{path}_lr.png", vmin=vmin, vmax=vmax)
                construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                             data=inv_visuals["INTERPOLATED"],
                                             path=f"{path}_interpolated.png", vmin=vmin, vmax=vmax)
                construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                             data=construct_mask(residuals),
                                             path=f"{path}_residual.png", vmin=-1, vmax=1,
                                             costline_color="red", cmap="binary",
                                             label="Signum(SR - HR)")
                construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                             data=mean_candidate,
                                             path=f"{path}_mean_sr.png", vmin=vmin, vmax=vmax)
                construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                             data=std_candidate,
                                             path=f"{path}_std_sr.png", vmin=0.0, cmap="Greens")

        normalizing_constant = idx // configs.val_vis_freq
        val_metrics["MSE"] /= normalizing_constant
        val_metrics["RMSE"] /= normalizing_constant
        val_metrics["MR"] /= normalizing_constant
        val_metrics["MAE"] /= normalizing_constant
        val_metrics["mean_bias_over_pixels"] /= normalizing_constant
        val_metrics["std_bias_over_pixels"] /= normalizing_constant

        message = f"Epoch: {current_epoch:5}  |  Iteration: {current_step:8}"
        for metric, value in val_metrics.items():
            message = f"{message}  |  {metric:s}: {value:.5f}"
        val_logger.info(message)

    torch.save(inv_visuals["LR"], f"{path}_LR.pt")
    torch.save(inv_visuals["HR"], f"{path}_HR.pt")
    torch.save(inv_visuals["INTERPOLATED"], f"{path}_INTERPOLATED.pt")
    torch.save(inv_visuals["SR"], f"{path}_SR.pt")
    torch.save(mean_candidate, f"{path}_mean_sr.pt")
    torch.save(std_candidate, f"{path}_std.pt")
    val_logger.info("End of testing.")
