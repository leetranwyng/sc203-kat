"""Small standard-library CLI; model loading lives in method profiles."""
import argparse
import json
from pathlib import Path
from .runtime.config import load_config, project_root

def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, list) else obj.get("records", [obj])

def main(argv=None):
    parser = argparse.ArgumentParser(prog="sc203-kat")
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("run", help="Run an isolated Kaggle method profile")
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true", help="Validate and show config, without GPU or installation")
    evaluate = subs.add_parser("evaluate", help="Score normalized prediction JSONL")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    visualize = subs.add_parser("visualize", help="Render a comparison HTML from normalized records")
    visualize.add_argument("--predictions", required=True)
    visualize.add_argument("--output", required=True)
    imp = subs.add_parser("import-results", help="Normalize the archived five-model result directory")
    imp.add_argument("--source", required=True)
    imp.add_argument("--output", required=True)
    imp.add_argument("--copy-images", action="store_true", help="Copy original images for a portable gallery")
    args = parser.parse_args(argv)
    if args.command == "run":
        root = project_root()
        config = load_config(args.config, root)
        if args.dry_run:
            from .methods import describe
            print(json.dumps(describe(config["method"], config, root=root), ensure_ascii=False, indent=2))
            return 0
        from .methods import run
        run(config["method"], config, root=root)
    elif args.command == "evaluate":
        from .evaluation.scoring import evaluate
        result = evaluate(read_records(args.predictions))
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(out)
    elif args.command == "visualize":
        from .analysis.html import render_report
        render_report(read_records(args.predictions), args.output)
        print(args.output)
    elif args.command == "import-results":
        from .analysis.import_existing import import_directory
        rows = import_directory(args.source, args.output, copy_images=args.copy_images)
        print(f"Imported {len(rows)} predictions into {args.output}")
    return 0
