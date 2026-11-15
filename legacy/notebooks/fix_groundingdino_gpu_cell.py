# Dan vao Code cell moi trong cung session Kaggle, roi chay cell nay.
import ast
from pathlib import Path

path = REPO / 'apc/vision_modules/detection.py'
source = path.read_text(encoding='utf-8')
tree = ast.parse(source)
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run_detection')
call = next(n for n in ast.walk(method) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
            and n.value.func.id == 'predict')
guarded = any(isinstance(n, ast.With) and any(x is call for child in n.body for x in ast.walk(child))
              for n in ast.walk(method))
if not guarded:
    lines = source.splitlines(keepends=True)
    block = lines[call.lineno - 1:call.end_lineno]
    lines[call.lineno - 1:call.end_lineno] = [
        '        with torch.cuda.device(self.device):\n'
    ] + ['    ' + line for line in block]
    patched = ''.join(lines)
    ast.parse(patched)
    backup = path.with_name(path.name + '.before_cuda_guard')
    if not backup.exists():
        backup.write_text(source, encoding='utf-8')
    path.write_text(patched, encoding='utf-8')
    print('Patched: GroundingDINO predict chay trong CUDA device context cua vision GPU.')
else:
    print('CUDA device guard da co.')

# launch tao worker moi, khong tai su dung CUDA context da loi.
ENV['CUDA_LAUNCH_BLOCKING'] = '1'
launch(smoke=True)
