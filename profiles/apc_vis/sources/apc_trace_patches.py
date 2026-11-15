"""Apply trace edits to the copied APC source after its compatibility patches."""

import ast
import hashlib
import json
from pathlib import Path
import sys


EDITS = {
    "apc/apc_pipeline.py": [
        ("import logging", "import logging\nimport json\nfrom pathlib import Path"),
        ("for obj_name in objects_of_interest:\n            self.logger.info",
         "for obj_index,obj_name in enumerate(objects_of_interest):\n"
         "            trace=Path(apc_args['trace_save_dir'])\n"
         "            prefix=trace / f'object_{obj_index}'\n"
         "            prefix.with_suffix('.name').write_text(obj_name)\n            self.logger.info"),
        ("if len(boxes) == 0:",
         "np.save(str(prefix)+'_boxes.npy', np.asarray([np.asarray(b.cpu() if hasattr(b,'cpu') else b) for b in boxes]))\n"
         "            if len(boxes) == 0:"),
        ("# largest mask",
         "# largest mask\n"
         "            Image.fromarray(np.asarray(segment_mask.cpu() if hasattr(segment_mask,'cpu') else segment_mask).astype('uint8')*255).save(str(prefix)+'_mask.png')\n"
         "            np.save(str(prefix)+'_selected_box.npy',np.asarray(box2D))"),
        ("return abstract_scene_tgt",
         "Path(os.environ['APC_TRACE_DIR'],'scene.json').write_text(json.dumps(dict(viewer=ref_viewer,camera=abstract_scene_src,transformed=abstract_scene_tgt),default=lambda x:np.asarray(x).tolist()))\n"
         "        return abstract_scene_tgt"),
    ],
    "apc/vision_modules/detection.py": [
        ("        boxes_sorted = sort_by_scores",
         "        from pathlib import Path\n        import json\n"
         "        with (Path(os.environ['APC_TRACE_DIR'])/'detector.jsonl').open('a') as stream:\n"
         "            stream.write(json.dumps(dict(category=category,boxes=boxes.detach().cpu().tolist(),scores=scores.detach().cpu().tolist()))+'\\n')\n"
         "        boxes_sorted = sort_by_scores"),
    ],
    "apc/vision_modules/orientation.py": [
        ("        # Obtain each angle",
         "        from pathlib import Path\n        import json\n"
         "        folder=Path(apc_args['trace_save_dir'])\n"
         "        record={'category':category,'bbox':list(bbox) if bbox is not None else None,'angles':np.asarray(pred_angles).tolist()}\n"
         "        index=len(list(folder.glob('orientation_input_*.png')))\n"
         "        image.save(folder/f'orientation_input_{index:02d}.png')\n"
         "        # Obtain each angle"),
        ("        direction_vector = self.orientation_to_direction(azimuth_deg, polar_deg)",
         "        direction_vector = self.orientation_to_direction(azimuth_deg, polar_deg)\n"
         "        record['direction_vector']=np.asarray(direction_vector).tolist()\n"
         "        (folder/f'orientation_angles_{index:02d}.json').write_text(json.dumps(record,default=lambda x:np.asarray(x).tolist()))"),
    ],
}


def apply_patches(repo):
    changes = []
    for name, edits in EDITS.items():
        path = Path(repo)/name
        source = path.read_text(encoding="utf-8")
        original = source
        for before, after in edits:
            if after in source:
                continue
            if source.count(before) != 1:
                raise ValueError(f"Expected exactly one trace patch target in {name}: {before[:100]}")
            source = source.replace(before, after, 1)
        ast.parse(source)
        path.write_text(source, encoding="utf-8")
        changes.append(dict(file=name, before_sha=hashlib.sha256(original.encode()).hexdigest(),
                            after_sha=hashlib.sha256(source.encode()).hexdigest()))
    return changes


if __name__ == "__main__":
    changes = apply_patches(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(changes, indent=2), encoding="utf-8")
