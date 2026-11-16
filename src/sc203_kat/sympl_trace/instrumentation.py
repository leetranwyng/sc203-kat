from pathlib import Path

def patch_source(files, name, old, new):
    source = files[name].decode('utf8').replace('\r\n', '\n')
    assert source.count(old) == 1, (name, old[:70], source.count(old))
    files[name] = source.replace(old, new, 1).encode('utf8')

def instrument(repo):
    repo=Path(repo)
    files={"repo/"+p.relative_to(repo).as_posix():p.read_bytes() for p in (repo/"sympl").rglob("*.py")}
    patch_source(files, 'repo/sympl/vision_modules/depth.py', '        xs, ys, zs = xs[keep], ys[keep], zs[keep]', "        from trace_runtime import snapshot\n        snapshot('depth_band', dict(bbox=list(bbox), xs=xs, ys=ys, zs=zs, keep=keep,\n                 hist=hist, bin_edges=bin_edges, z_mode=z_mode, band_param=band_param,\n                 depth_band=depth_band, use_center_weight=use_center_weight, K=K))\n        xs, ys, zs = xs[keep], ys[keep], zs[keep]")
    patch_source(files, 'repo/sympl/vision_modules/depth.py', '        return np.array([x_final, y_final, z_final])', "        snapshot('depth_center', dict(xs=xs,ys=ys,zs=zs,w=w,X=X,Y=Y,\n                 weighted_mean=[xw,yw,zw], median=[x_med,y_med,z_med], alpha=alpha,\n                 position=[x_final,y_final,z_final]))\n        return np.array([x_final, y_final, z_final])")
    patch_source(files, 'repo/sympl/projection.py', '    return u, v, meta\n', "    from trace_runtime import snapshot\n    snapshot('projection_parameters',dict(points_cam=points_cam,K=K,ref_position=ref_position,\n             image_size=image_size,scale_s=s,ref_uv=ref_uv,u=u,v=v,meta=meta))\n    return u, v, meta\n")
    patch_source(files, 'repo/sympl/utils.py', '    def wrapper(*args, **kwargs):', "    @__import__('functools').wraps(func)\n    def wrapper(*args, **kwargs):")
    patch_source(files, 'repo/sympl/vision_modules/detection.py', '        return boxesAbs[selected_idx], conv_history', "        from trace_runtime import snapshot\n        matched=[i for i in range(min(self.num_candidates,len(boxesAbs))) if str(i) in str(response)]\n        snapshot('refinement_selection',dict(category=category,response=response,selected_idx=selected_idx,\n                 matched_indices=matched,fallback_used=not bool(matched),boxes_pixel=boxesAbs))\n        return boxesAbs[selected_idx], conv_history")
    for name, data in files.items():
        (repo/name.removeprefix("repo/")).write_bytes(data)
