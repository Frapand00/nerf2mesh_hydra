class MyCallback:
    def on_train_epoch_end(self):

        # the new inplace TV loss
        if self.stage == 0 and self.lambda_tv > 0:

            # # progressive...
            # lambda_tv = min(1.0, self.global_step / 10000) * self.lambda_tv
            lambda_tv = self.lambda_tv

            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)

            # different tv weights for inner and outer points
            if self.bound > 1:
                mask_inner = self.tmp_xyzs.abs().amax(dim=-1) <= 1
                xyzs_inner = self.tmp_xyzs[mask_inner].contiguous()
                xyzs_outer = self.tmp_xyzs[~mask_inner].contiguous()

                self.model.encoder.grad_total_variation(lambda_tv, xyzs_inner, self.model.bound)
                self.model.encoder.grad_total_variation(lambda_tv * 10, xyzs_outer, self.model.bound)
            else:
                self.model.encoder.grad_total_variation(lambda_tv, self.tmp_xyzs, self.model.bound)
