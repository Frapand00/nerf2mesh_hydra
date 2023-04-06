import argparse
import torch
from typing import Optional
import pyrootutils

from omegaconf import DictConfig
import hydra
import os
print(torch.cuda.is_available())
from nerf.gui import NeRFGUI
from nerf.network import NeRFNetwork
from nerf.utils import *
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# torch.autograd.set_detect_anomaly(True)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print(cfg.model)
    ### training options
    #parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    
    if cfg.model.contract:
        # mark untrained is not very correct in contraction mode...
        cfg.trainer.mark_untrained = False

    if cfg.data.enable_sparse_depth:
        print(f'[WARN] disable random image batch when depth supervision is used!')
        cfg.data.random_image_batch = False
    
    if cfg.data.patch_size > 1:
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert cfg.general.num_rays % (cfg.data.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    '''    if cfg.general.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif cfg.general.data_format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    else: # 'nerf
        from nerf.provider import NeRFDataset
    '''
    # convert ratio to steps
    cfg.trainer.refine_steps = [int(round(x * cfg.general.iters)) for x in cfg.general.refine_steps_ratio]

    seed_everything(cfg.general.seed)


    model = hydra.utils.instantiate(cfg.model)
    #print(model)
    
    # criterion = torch.nn.MSELoss(reduction='none')
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if cfg.general.test:
        
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if cfg.general.gui:
            gui = hydra.utils.instantiate(cfg.gui, trainer= trainer)
            gui.render()
        
        else:
            if not cfg.general.test_no_video:
                cfg.data.type = 'test'
                test_loader = hydra.utils.instantiate(cfg.data).dataloader()

                if test_loader.has_gt:
                    trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)] # set up metrics
                    trainer.evaluate(test_loader) # blender has gt, so evaluate it.

                trainer.test(test_loader, write_video=True) # test and save video
            
            if not cfg.general.test_no_mesh:
                if cfg.general.stage == 1:
                    trainer.export_stage1(resolution=cfg.general.texture_size)
                else:
                    # need train loader to get camera poses for visibility test
                    if cfg.general.mesh_visibility_culling:
                        cfg.data.type = 'train'
                        train_loader = hydra.utils.instantiate(cfg.data).dataloader()
                    trainer.save_mesh(resolution=512, decimate_target=3e5, dataset=train_loader._data if cfg.general.mesh_visibility_culling else None)
        
    else:
        
        optimizer = lambda model: torch.optim.Adam(model.get_params(cfg.model.lr), eps=1e-15)

        train_loader = hydra.utils.instantiate(cfg.data).dataloader()

        max_epoch = np.ceil(cfg.general.iters / len(train_loader)).astype(np.int32)
        save_interval = max(1, max_epoch // 50) # save ~50 times during the training
        eval_interval = max(1, max_epoch // 10)
        #update cfg trainer
        cfg.trainer.eval_interval = int(eval_interval)
        cfg.trainer.save_interval = int(save_interval)
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}.')

        if cfg.model.ind_dim > 0:
            assert len(train_loader) < cfg.model.ind_num, f"[ERROR] dataset too many frames: {len(train_loader)}, please increase --ind_num to at least this number!"

        # colmap can estimate a more compact AABB
        if cfg.general.data_format == 'colmap':
            model.update_aabb(train_loader._data.pts_aabb)

        # scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.iters // 2, opt.iters * 3 // 4, opt.iters * 9 // 10], gamma=0.33)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / cfg.general.iters, 1))

        trainer = hydra.utils.instantiate(cfg.trainer, model = model, optimizer=optimizer, criterion=criterion,
                          ema_decay=0.95 if cfg.general.stage == 0 else None,
                          lr_scheduler=scheduler )

        '''        if cfg.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()'''
        if cfg.model.ind_dim == 'dgsdgsdbdfh':
            print()
        else:
            cfg.data.type = 'val'
            valid_loader = hydra.utils.instantiate(cfg.data).dataloader()

            trainer.metrics = [PSNRMeter(),]
            trainer.train(train_loader, valid_loader, max_epoch)
            
            # last validation
            trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
            trainer.evaluate(valid_loader)

            # also test
            cfg.data.type = 'test'
            test_loader = hydra.utils.instantiate(cfg.data).dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video
            
            if cfg.general.stage == 1:
                trainer.export_stage1(resolution=cfg.general.texture_size)
            else:
                trainer.save_mesh(resolution=512, decimate_target=3e5, dataset=train_loader._data if cfg.general.mesh_visibility_culling else None)


if __name__ == "__main__":
    main()