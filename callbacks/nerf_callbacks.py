class MyCallback:
    def on_train_epoch_end(self):

        # the new inplace TV loss
        if self.opt.lambda_tv > 0:
            # # progressive...
            # lambda_tv = min(1.0, self.global_step / 10000) * self.opt.lambda_tv
            lambda_tv = self.opt.lambda_tv

            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)

            # different tv weights for inner and outer points
            self.model.apply_total_variation(lambda_tv)