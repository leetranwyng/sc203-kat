"""Use APCExperiment with cached depth/detections and only required live tools."""
def make(cfg, context, callback):
    import torch
    from types import SimpleNamespace
    from torchvision.transforms import Compose
    from apc import apc_pipeline
    from segment_anything import SamPredictor, sam_model_registry
    from apc_experiment import APCExperiment
    originals=[]
    def patch(cls, fn):
        originals.append((cls,cls.__init__));cls.__init__=fn
    def detection_init(self,config,device='cpu'):
        self.device=device;self.config=config;self.num_candidates=config.detection.num_candidates
        self.detection_model=torch.nn.Identity()
        self.segmentation_model=(sam_model_registry['default'](checkpoint=config.segmentation.ckpt_path)
                                 if cfg['need_sam1'] else torch.nn.Identity()).to(device)
        if not cfg['need_sam1']:
            self.segmentation_model.image_format='RGB'
            self.segmentation_model.image_encoder=SimpleNamespace(img_size=1024)
        self.segmentation_predictor=SamPredictor(self.segmentation_model)
    def depth_init(self,config,device='cpu'):
        self.device=device;self.config=config;self.depth_model=torch.nn.Identity()
        self.depth_transform=Compose([torch.nn.Identity() for _ in range(4)])
    def orientation_init(self,config,device='cpu'):
        self.device=device;self.config=config;self.orientation_model=torch.nn.Identity()
    patch(apc_pipeline.DetectionModule,detection_init)
    patch(apc_pipeline.DepthModule,depth_init)
    if not cfg['need_oa1']:patch(apc_pipeline.OrientationModule,orientation_init)
    try:
        return APCExperiment(context['cfg'],{'root':context['asset_root']},callback,sanity=cfg['mode']=='sanity')
    finally:
        for cls,fn in reversed(originals):cls.__init__=fn
