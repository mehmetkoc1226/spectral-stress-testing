"""Verify archived evidence and numerical implementations without GPU inference."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, tempfile, shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def call(args,cwd=ROOT):
    print('Running:', ' '.join(map(str,args)),flush=True)
    result=subprocess.run(list(map(str,args)),cwd=cwd,capture_output=True,text=True)
    print(result.stdout,flush=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-pytest',action='store_true',help='Use when the complete suite has already run, e.g. in CI.')
    args=parser.parse_args()
    record={'gpu_inference_executed':False,'new_empirical_results':False}
    if not args.skip_pytest:
        call([sys.executable,'-m','pytest','-q','tests/test_configs.py','tests/test_frozen_artifacts.py','tests/test_repository_language.py'])
        record['lightweight_repository_tests']=6
    cpu=json.loads(call([sys.executable,'strengthening_experiments/cpu_checks.py']))
    assert cpu['status']=='passed';record['numerical_checks']=cpu['n_checks']
    s2=ROOT/'artifacts/frozen_runs/stage2/spectral_stage2_calibration_pilot_v1__27bb9eee96fb'
    replay=ROOT/'analysis_results/provenance/stage2'
    files=['config.json','fit_parameters/resnet50_a1.json','fit_parameters/swin_t_in1k.json','fit_parameters/vitb16_in1k.json']
    for name in files:assert json.loads((s2/name).read_text())==json.loads((replay/name).read_text()),name
    for name in ['calibration_split_manifest.csv','dataset_manifest_imagenette.csv','dataset_manifest_imagenet_a.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(s2/name),pd.read_csv(replay/name))
    s1=ROOT/'artifacts/frozen_runs/stage1/spectral_stage1_confirmatory_v1__7044005bc7a4/summary.csv'
    pd.testing.assert_frame_equal(pd.read_csv(s1),pd.read_csv(ROOT/'analysis_results/provenance/stage1_summary.csv'))
    record['original_and_followup_inputs_agree']=True
    with tempfile.TemporaryDirectory() as directory:
        temp=Path(directory)
        for folder in ['strengthening_experiments','analysis_results','assets']:
            shutil.copytree(ROOT/folder,temp/folder,ignore=shutil.ignore_patterns('__pycache__'))
        (temp/'figures').mkdir()
        result=json.loads(call([sys.executable,'assets/build_strengthening_results.py'],cwd=temp))
        assert result['status']=='passed';record['saved_results']=result
    # Check distributed file identity; generated validation records are outside this manifest.
    count=0
    for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        count+=1
    record['file_hashes_verified']=count
    record['status']='passed'
    out=ROOT/'validation/REPRODUCIBILITY_CHECK.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record,indent=2)+'\n')
    print('PASS: saved evidence and CPU checks verified. Record:',out,flush=True)

if __name__=='__main__':main()
