import lightning as L


class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ...

    def training_step(self, batch, batch_idx):
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y, y_hat)
        acc = self.accuracy(y, y_hat)
        ...
        return loss

    def training_step(self, batch):

        rays_o = batch['rays_o']  # [N, 3]
        rays_d = batch['rays_d']  # [N, 3]
        index = batch['index']  # [1/N]
        cam_near_far = batch['cam_near_far'] if 'cam_near_far' in batch else None  # [1/N, 2] or None

        images = batch['images']  # [N, 3/4]

        N, C = images.shape

        if self.opt.background == 'random':
            bg_color = torch.rand(N, 3, device=self.device)  # [N, 3], pixel-wise random.
        else:  # white / last_sample
            bg_color = 1

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        shading = 'diffuse' if self.global_step < self.opt.diffuse_step else 'full'
        update_proposal = self.global_step <= 3000 or self.global_step % 5 == 0

        outputs = self.model.render(rays_o, rays_d, index=index, bg_color=bg_color, perturb=True,
                                    cam_near_far=cam_near_far, shading=shading, update_proposal=update_proposal)

        # MSE loss
        pred_rgb = outputs['image']
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)  # [N, 3] --> [N]

        loss = loss.mean()

        # extra loss
        if 'proposal_loss' in outputs and self.opt.lambda_proposal > 0:
            loss = loss + self.opt.lambda_proposal * outputs['proposal_loss']

        if 'distort_loss' in outputs and self.opt.lambda_distort > 0:
            loss = loss + self.opt.lambda_distort * outputs['distort_loss']

        if self.opt.lambda_entropy > 0:
            w = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            entropy = - w * torch.log2(w) - (1 - w) * torch.log2(1 - w)
            loss = loss + self.opt.lambda_entropy * (entropy.mean())

        # adaptive num_rays
        if self.opt.adaptive_num_rays:
            self.opt.num_rays = int(round((self.opt.num_points / outputs['num_points']) * self.opt.num_rays))

        return pred_rgb, gt_rgb, loss


    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            preds, truths, loss_net = self.train_step(data)

            loss = loss_net

            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss_net.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.6f} ({total_loss / self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss / self.local_step:.6f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}, loss={average_loss:.6f}.")


    def configure_optimizers(self):
        # Return one or several optimizers
        return torch.optim.Adam(self.parameters(), ...)

    def train_dataloader(self):
        # Return your dataloader for training
        return DataLoader(...)

    def on_train_start(self):
        # Do something at the beginning of training
        ...

    def any_hook_you_like(self, *args, **kwargs):
        ...