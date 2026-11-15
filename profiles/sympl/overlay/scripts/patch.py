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
    # External sources exist after HTTPS cloning.
    edit('sympl/vision_modules/src/orient_anything/paths.py',lambda s:s.replace('"facebook/dinov2-large"',repr(str(assets/'dinov2_large'))))
    edit('sympl/vision_modules/src/orient_anything/vision_tower.py',lambda s:s.replace('bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:', 'bool_masked_pos: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:').replace('embeddings = self.dropout(embeddings)',"embeddings = getattr(self, 'dropout', torch.nn.Identity())(embeddings)"))
    edit('sympl/vision_modules/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',lambda s:s.replace('"bert-base-uncased"',repr(str(assets/'bert_base_uncased'))).replace("'bert-base-uncased'",repr(str(assets/'bert_base_uncased'))))
    return changes

if __name__=='__main__':
    import sys
    changes=patch_repo(sys.argv[1],sys.argv[2]);Path(sys.argv[3]).write_text(json.dumps(changes,indent=2),encoding='utf-8')
