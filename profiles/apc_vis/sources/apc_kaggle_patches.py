"""Apply auditable Kaggle compatibility fixes to a copied APC snapshot, not the checkout."""
import ast,hashlib,json,os
from pathlib import Path

def patch_repo(repo,assets):
    repo=Path(repo);assets=Path(assets);changes=[]
    def edit(rel,fn):
        p=repo/rel
        backup=p.with_name(p.name+'.apc_original')
        if not backup.exists():backup.write_text(p.read_text(),encoding='utf-8')
        old=backup.read_text();new=fn(old)
        if new!=old:
            ast.parse(new);p.write_text(new,encoding='utf-8')
            changes.append(dict(file=rel,before_sha=hashlib.sha256(old.encode()).hexdigest(),after_sha=hashlib.sha256(new.encode()).hexdigest()))
    def detection(s):
        s=s.replace('load_model(config_path, config.detection.ckpt_path)','load_model(config_path, config.detection.ckpt_path, device=str(self.device))')
        needle='            text_threshold=self.config.detection.text_threshold,\n'
        assert needle in s;s=s.replace(needle,needle+'            device=str(self.device),\n')
        # GroundingDINO's custom CUDA attention uses getCurrentCUDAStream()
        # without a device guard. Match the current GPU to its input tensors.
        tree=ast.parse(s)
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_detection')
        call=next(n for n in ast.walk(method) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='predict')
        lines=s.splitlines(keepends=True)
        block=lines[call.lineno-1:call.end_lineno]
        lines[call.lineno-1:call.end_lineno]=['        with torch.cuda.device(self.device):\n']+['    '+line for line in block]
        s=''.join(lines)
        return s
    edit('apc/vision_modules/detection.py',detection)
    def orientation(s):
        s=s.replace('from .src.orient_anything.render import render, Model','')
        needle='        axis_model = Model(axis_obj_path, texture_filename=axis_texture_path)'
        s=s.replace(needle,'        from .src.orient_anything.render import render, Model\n'+needle)
        tree=ast.parse(s)
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='orientation_ckpt_path' for t in n.targets))
        s=s.replace(ast.get_source_segment(s,assignment),'orientation_ckpt_path = str(config.orientation.model_path)',1)
        s=s.replace('"facebook/dinov2-large",','str(config.orientation.processor_path),')
        return s
    edit('apc/vision_modules/orientation.py',orientation)
    def renderer(s):
        s=s.replace("        if apc_args['render_whole_scene']:","        z_trans = 0.0  # No backward shift needed when all objects are already in front.\n        if apc_args['render_whole_scene']:")
        s=s.replace('        # Normalize positions','        orientations = [orientations[i] for i in positions_ori_idx]\n        colors = [colors[i] for i in positions_ori_idx]\n\n        # Normalize positions')
        s=s.replace('            xy_scale = grid_size / max_xy','            xy_scale = grid_size / max(max_xy, 1e-6)')
        s=s.replace("            z_list = [pos[2] for pos in positions]\n            max_z_obj_idx", "            if not positions:\n                raise ValueError('No objects to render after excluding reference viewer')\n            z_list = [pos[2] for pos in positions]\n            max_z_obj_idx",1)
        return s
    edit('apc/renderer.py',renderer)
    # Keep the mask statistics/geometry method; only remove a slow Python pixel loop.
    def depth(s):
        a=s.index('        depth_values = []',s.index('def unproject_to_3D'))
        b=s.index('        # Find the most frequent depth',a)
        s=s[:a]+"        depth_values = depth_unproj[ys,xs]\n        if depth_values.size == 0:\n            raise ValueError('Empty segmentation mask; no 3D points')\n\n"+s[b:]
        a=s.index('        points_filtered = []',s.index('def unproject_to_3D'))
        b=s.index('        # Compute final 3D coordinates',a)
        s=s[:a]+"        z=depth_unproj[ys,xs][depth_values_filtered]\n        x=(xs[depth_values_filtered]-K[0,2])*z/K[0,0]\n        y=(ys[depth_values_filtered]-K[1,2])*z/K[1,1]\n        points_filtered=np.column_stack((x,-y,-z))\n        if not len(points_filtered) or not np.isfinite(points_filtered).all():\n            raise ValueError('Empty/nonfinite filtered 3D points')\n\n"+s[b:]
        return s
    edit('apc/vision_modules/depth.py',depth)
    # External sources exist after HTTPS cloning.
    edit('apc/vision_modules/src/orient_anything/paths.py',lambda s:s.replace('"facebook/dinov2-large"',repr(str(assets/'dinov2_large'))))
    edit('apc/vision_modules/src/orient_anything/vision_tower.py',lambda s:s.replace('bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:', 'bool_masked_pos: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:').replace('embeddings = self.dropout(embeddings)',"embeddings = getattr(self, 'dropout', torch.nn.Identity())(embeddings)"))
    edit('apc/vision_modules/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',lambda s:s.replace('"bert-base-uncased"',repr(str(assets/'bert_base_uncased'))).replace("'bert-base-uncased'",repr(str(assets/'bert_base_uncased'))))
    return changes

if __name__=='__main__':
    import sys
    changes=patch_repo(sys.argv[1],sys.argv[2]);Path(sys.argv[3]).write_text(json.dumps(changes,indent=2),encoding='utf-8')
