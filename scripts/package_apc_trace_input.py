"""Package or repair trace DATA as a Kaggle-safe ZIP, without expanded duplicates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sc203_kat.ablation.bundle import package,repair_zip

if __name__=='__main__':
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',help='Original APC replay folder')
    source.add_argument('--repair-zip',help='Existing trace bundle ZIP')
    p.add_argument('--output',required=True,help='Destination ZIP (or stem without .zip)')
    a=p.parse_args()
    result=repair_zip(a.repair_zip,a.output) if a.repair_zip else package(a.source,a.output)
    print(json.dumps(result,indent=2))
