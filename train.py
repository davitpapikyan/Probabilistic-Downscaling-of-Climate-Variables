"""The training script for DDPM model.
"""
import argparse
import logging
import os
import pickle
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from aim import Run, num_utils
from tensorboardX import SummaryWriter
from torch.nn.functional import mse_loss, l1_loss
from torch.utils.data import DataLoader

import model
from config import Config
from utils import dict2str, setup_logger, construct_and_save_wbd_plots, \
    accumulate_statistics, get_transformation, \
    get_optimizer, construct_mask, reverse_transform_candidates, set_seeds
from weatherbench_data import collate_wb_batch, create_datasets, create_dataloaders
from weatherbench_data.utils import reverse_transform, reverse_transform_tensor, prepare_test_data

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

    setup_logger(None, configs.log, "train", screen=True)
    setup_logger("val", configs.log, "val")
    logger = logging.getLogger("base")
    val_logger = logging.getLogger("val")
    logger.info(dict2str(configs.get_hyperparameters_as_dict()))
    tb_logger = SummaryWriter(log_dir=configs.tb_logger)

    aim_logger = Run(run_hash=configs.name, repo='./experiments/aim/', experiment=configs.name)
    aim_logger["hparams"] = {"train_min_date": configs.train_min_date, "train_max_date": configs.train_max_date,
                             "val_min_date": configs.val_min_date, "val_max_date": configs.val_max_date,
                             "variables": configs.variables, "transformation": configs.transformation,
                             "tranform_monthly": configs.tranform_monthly, "batch_size": configs.batch_size,
                             "norm_groups": configs.norm_groups, "dropout": configs.dropout,
                             "diffusion_loss": configs.diffusion_loss, "init_method": configs.init_method,
                             "train_schedule": configs.train_schedule, "val_schedule": configs.val_schedule,
                             "optimizer": configs.optimizer_type, "learning_rate": configs.lr}

    transformation = get_transformation(configs.transformation)
    train_data, val_data, metadata, transformations = create_datasets(dataroot=configs.dataroot,
                                                                      name=configs.name,
                                                                      train_min_date=configs.train_min_date,
                                                                      train_max_date=configs.train_max_date,
                                                                      val_min_date=configs.val_min_date,
                                                                      val_max_date=configs.val_max_date,
                                                                      variables=configs.variables,
                                                                      transformation=transformation,
                                                                      storage_root=configs.experiments_root,
                                                                      apply_tranform_monthly=configs.tranform_monthly)
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}.")
    train_loader, val_loader = create_dataloaders(train_data, val_data, batch_size=configs.batch_size,
                                                  use_shuffle=configs.use_shuffle, num_workers=configs.num_workers)
    logger.info("Training and Validation dataloaders are ready.")

    # Defining the model.
    optimizer = get_optimizer(configs.optimizer_type)
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
                                   finetune_norm=configs.finetune_norm, optimizer=optimizer, amsgrad=configs.amsgrad,
                                   learning_rate=configs.lr, checkpoint=configs.checkpoint,
                                   resume_state=configs.resume_state, phase=configs.phase, height=configs.height)
    logger.info("Model initialization is finished.")

    current_step, current_epoch = diffusion.begin_step, diffusion.begin_epoch
    if configs.resume_state:
        logger.info(f"Resuming training from epoch: {current_epoch}, iter: {current_step}.")

    logger.info("Starting the training.")
    diffusion.set_new_noise_schedule(schedule=configs.train_schedule, n_timestep=configs.train_n_timestep,
                                     linear_start=configs.train_linear_start, linear_end=configs.train_linear_end)

    accumulated_statistics = OrderedDict()

    # Creating placeholder for storing validation metrics Mean Squared Error, Root MSE, Mean Residual.
    val_metrics = OrderedDict({"MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "MR": 0.0})

    # Training.
    while current_step < configs.n_iter:
        current_epoch += 1

        for train_data in train_loader:
            current_step += 1

            if current_step > configs.n_iter:
                break

            # Training.
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            # diffusion.lr_scheduler_step()  # For lr scheduler updates per iteration. 
            accumulate_statistics(diffusion.get_current_log(), accumulated_statistics)

            # Logging the training information.
            if current_step % configs.print_freq == 0:
                message = f"Epoch: {current_epoch:5}  |  Iteration: {current_step:8}"

                for metric, values in accumulated_statistics.items():
                    mean_value = np.mean(values)
                    message = f"{message}  |  {metric:s}: {mean_value:.5f}"
                    tb_logger.add_scalar(f"{metric}/train", mean_value, current_step)
                    aim_logger.track(num_utils.convert_to_py_number(mean_value), name=metric, step=current_step,
                                     epoch=current_epoch, context={"subset": "train"})

                logger.info(message)
                # tb_logger.add_scalar(f"learning_rate", diffusion.get_lr(), current_step)

                # Visualizing distributions of parameters.
                for name, param in diffusion.get_named_parameters():
                    tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), current_step)

                accumulated_statistics = OrderedDict()

            # Validation.
            if current_step % configs.val_freq == 0:
                logger.info("Starting validation.")
                idx = 0
                result_path = f"{configs.results}/{current_epoch}"
                os.makedirs(result_path, exist_ok=True)
                diffusion.set_new_noise_schedule(schedule=configs.val_schedule,
                                                 n_timestep=configs.val_n_timestep,
                                                 linear_start=configs.val_linear_start,
                                                 linear_end=configs.val_linear_end)

                # A dictionary for storing a list of mean temperatures for each month.
                month2mean_temperature = defaultdict(list)

                for val_data in val_loader:
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continuous=False)  # Continues=False to return only the last timesteps's outcome.

                    # Computing metrics on vlaidation data.
                    visuals = diffusion.get_current_visuals()

                    inv_visuals = reverse_transform(visuals, transformations,
                                                    configs.variables, diffusion.get_months(),
                                                    configs.tranform_monthly)

                    # Computing MSE and RMSE on original data.
                    mse_value = mse_loss(inv_visuals["HR"], inv_visuals["SR"])
                    val_metrics["MSE"] += mse_value
                    val_metrics["RMSE"] += torch.sqrt(mse_value)
                    val_metrics["MAE"] += l1_loss(inv_visuals["HR"], inv_visuals["SR"])

                    mean_temp_pred = inv_visuals["SR"].mean(axis=[1, 2, 3])
                    for m, t in zip(diffusion.get_months(), mean_temp_pred):
                        month2mean_temperature[int(m)].append(t)

                    # Computing residuals for visualization.
                    residuals = inv_visuals["SR"] - inv_visuals["HR"]
                    val_metrics["MR"] += residuals.mean()

                    if idx % configs.val_vis_freq == 0:
                        path = f"{result_path}/{current_epoch}_{current_step}_{idx}"
                        logger.info(f"[{idx//configs.val_vis_freq}] Visualizing and storing some examples.")

                        sr_candidates = diffusion.generate_multiple_candidates(n=configs.sample_size)
                        reverse_transform_candidates(sr_candidates, reverse_transform_tensor,
                                                     transformations, configs.variables,
                                                     "hr", diffusion.get_months(),
                                                     configs.tranform_monthly)
                        mean_candidate = sr_candidates.mean(dim=0)  # [B, C, H, W]
                        std_candidate = sr_candidates.std(dim=0)  # [B, C, H, W]
                        bias = mean_candidate - inv_visuals["HR"]
                        mean_bias_over_pixels = bias.mean()  # Scalar.
                        std_bias_over_pixels = bias.std()  # Scalar.

                        # Computing min and max measures to set a fixed colorbar for all visualizations.
                        vmin = min(inv_visuals["HR"][:configs.n_val_vis].min(),
                                   inv_visuals["SR"][:configs.n_val_vis].min(),
                                   inv_visuals["LR"][:configs.n_val_vis].min(),
                                   inv_visuals["INTERPOLATED"][:configs.n_val_vis].min(),
                                   mean_candidate[:configs.n_val_vis].min())
                        vmax = max(inv_visuals["HR"][:configs.n_val_vis].max(),
                                   inv_visuals["SR"][:configs.n_val_vis].max(),
                                   inv_visuals["LR"][:configs.n_val_vis].max(),
                                   inv_visuals["INTERPOLATED"][:configs.n_val_vis].max(),
                                   mean_candidate[:configs.n_val_vis].max())

                        # Choosing the first n_val_vis number of samples to visualize.
                        construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                                     data=inv_visuals["HR"][:configs.n_val_vis],
                                                     path=f"{path}_hr.png", vmin=vmin, vmax=vmax)
                        construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                                     data=inv_visuals["SR"][:configs.n_val_vis],
                                                     path=f"{path}_sr.png", vmin=vmin, vmax=vmax)
                        construct_and_save_wbd_plots(latitude=metadata.lr_lat, longitude=metadata.lr_lon,
                                                     data=inv_visuals["LR"][:configs.n_val_vis],
                                                     path=f"{path}_lr.png", vmin=vmin, vmax=vmax)
                        construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                                     data=inv_visuals["INTERPOLATED"][:configs.n_val_vis],
                                                     path=f"{path}_interpolated.png", vmin=vmin, vmax=vmax)
                        construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                                     data=construct_mask(residuals[:configs.n_val_vis]),
                                                     path=f"{path}_residual.png", vmin=-1, vmax=1,
                                                     costline_color="red", cmap="binary",
                                                     label="Signum(SR - HR)")
                        construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                                     data=mean_candidate[:configs.n_val_vis],
                                                     path=f"{path}_mean_sr.png", vmin=vmin, vmax=vmax)
                        construct_and_save_wbd_plots(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                                                     data=std_candidate[:configs.n_val_vis],
                                                     path=f"{path}_std_sr.png", vmin=0.0, cmap="Greens")
                        
                        # tb_logger.add_scalar(f"mean_bias_over_pixels/val", mean_bias_over_pixels, current_step)
                        # tb_logger.add_scalar(f"std_bias_over_pixels/val", std_bias_over_pixels, current_step)
                        
                        aim_logger.track(num_utils.convert_to_py_number(mean_bias_over_pixels),
                                         name="mean_bias_over_pixels", step=current_step,
                                         epoch=current_epoch, context={"subset": "val"})
                        aim_logger.track(num_utils.convert_to_py_number(std_bias_over_pixels),
                                         name="std_bias_over_pixels", step=current_step,
                                         epoch=current_epoch, context={"subset": "val"})

                        # tb_figure = construct_tb_visualization(latitude=metadata.hr_lat, longitude=metadata.hr_lon,
                        #                                        data=(inv_visuals["INTERPOLATED"][-1].squeeze(),
                        #                                              inv_visuals["SR"][-1].squeeze(),
                        #                                              inv_visuals["HR"][-1].squeeze()))
                        # tb_logger.add_figure(tag=f"Iter_{current_epoch}_{current_step}",
                        #                      figure=tb_figure, global_step=idx)

                # Validation is finished.
                val_metrics["MSE"] /= idx
                val_metrics["RMSE"] /= idx
                val_metrics["MR"] /= idx
                val_metrics["MAE"] /= idx

                diffusion.set_new_noise_schedule(schedule=configs.train_schedule,
                                                 n_timestep=configs.train_n_timestep,
                                                 linear_start=configs.train_linear_start,
                                                 linear_end=configs.train_linear_end)

                message = f"Epoch: {current_epoch:5}  |  Iteration: {current_step:8}"
                for metric, value in val_metrics.items():
                    message = f"{message}  |  {metric:s}: {value:.5f}"
                    # tb_logger.add_scalar(f"{metric}/val", value, current_step)
                    aim_logger.track(num_utils.convert_to_py_number(value), name=metric, step=current_step,
                                     epoch=current_epoch, context={"subset": "val"})
                val_logger.info(message)

                val_metrics = val_metrics.fromkeys(val_metrics, 0.0)  # Sets all metrics to zero.

            if current_step % configs.save_checkpoint_freq == 0:
                logger.info("Saving models and training states.")
                diffusion.save_network(current_epoch, current_step)

            # Learning rate scheduler step per iteration.
            # diffusion.lr_scheduler_step()  # For lr scheduler updates per epoch.

    tb_logger.close()
    aim_logger.close()
    logger.info("End of training.")

    logger.info("Starting final evaluation on training set.")
    train_subset = prepare_test_data(variables=configs.variables, val_min_date=configs.train_subset_min_date,
                                     val_max_date=configs.train_subset_max_date, dataroot=configs.dataroot,
                                     transformations=transformations)
    train_loader = DataLoader(train_subset, batch_size=32,
                              collate_fn=collate_wb_batch,
                              pin_memory=True, num_workers=2)

    diffusion.set_new_noise_schedule(schedule=configs.val_schedule,
                                     n_timestep=configs.val_n_timestep,
                                     linear_start=configs.val_linear_start,
                                     linear_end=configs.val_linear_end)
    with torch.no_grad():
        idx = 0
        train_metrics = OrderedDict({"MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "MR": 0.0})
        for train_data in train_loader:
            idx += 1
            diffusion.feed_data(train_data)
            diffusion.test(continuous=False)
            visuals = diffusion.get_current_visuals()
            inv_visuals = reverse_transform(visuals, transformations,
                                            configs.variables, diffusion.get_months(),
                                            configs.tranform_monthly)
            mse_value = mse_loss(inv_visuals["HR"], inv_visuals["SR"])
            train_metrics["MSE"] += mse_value
            train_metrics["RMSE"] += torch.sqrt(mse_value)
            train_metrics["MR"] += (inv_visuals["SR"] - inv_visuals["HR"]).mean()
            train_metrics["MAE"] += l1_loss(inv_visuals["HR"], inv_visuals["SR"])

        train_metrics["MSE"] /= idx
        train_metrics["RMSE"] /= idx
        train_metrics["MR"] /= idx
        train_metrics["MAE"] /= idx

    message = f"Final evaluation on train set"
    for metric, value in train_metrics.items():
        message = f"{message}  |  {metric:s}: {value:.5f}"
    logger.info(message)

    with open(f"{result_path}/month2mean_temperature.pickle", 'wb') as handle:
        pickle.dump(month2mean_temperature, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("End of evaluation.")
