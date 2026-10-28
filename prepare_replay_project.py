"""Prepare frozen experiment inputs and optionally regenerate calibration logits.

No temperatures are refitted. Recorded manifests and fitting coefficients are
copied byte-for-byte; generated logits are written to a new project directory.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,json,shutil,sys

ROOT=Path(__file__).resolve().parent
ORIGINAL='spectral_stage2_calibration_pilot_v1__27bb9eee96fb'

def stage(project):
    source=ROOT/'analysis_results/provenance/stage2'
    required=['config.json','calibration_split_manifest.csv',
              'dataset_manifest_imagenette.csv','dataset_manifest_imagenet_a.csv',
              *['fit_parameters/'+alias+'.json' for alias in
                ['resnet50_a1','swin_t_in1k','vitb16_in1k']]]
    for name in required:
        if not (source/name).is_file():
            raise FileNotFoundError('Missing distributed input: '+str(source/name))
    dest=project/'runs'/ORIGINAL
    for p in source.rglob('*'):
        if not p.is_file():continue
        q=dest/p.relative_to(source)
        if q.exists():
            if q.read_bytes()!=p.read_bytes():
                raise RuntimeError('Existing frozen record differs: '+str(q))
        else:
            q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,q)
    print('Frozen configuration, manifests and fitted coefficients:',dest,flush=True)
    return dest

def regenerate(project,run,batch_size,workers):
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0,str(ROOT/'strengthening_experiments'))
    import spectral_stage2_pilot as pilot
    if not torch.cuda.is_available():
        raise RuntimeError('Calibration-logit regeneration requires a CUDA GPU.')
    config=json.loads((run/'config.json').read_text())
    pilot.seed_everything(config['seed'])
    pilot.RUN_PROFILE='paper'
    design=dataclasses.replace(pilot.build_design(),
        models=tuple(config['models']),seed=config['seed'],batch_size=batch_size,
        num_workers=workers,mixed_precision=config['mixed_precision'],
        calibration_conditions=tuple(config['calibration_conditions']))
    device=torch.device('cuda')
    logger=pilot.setup_logger(project/'regenerate_fit_logits.log')
    cache=Path('/content/cviu_fit_cache') if Path('/content').is_dir() else project/'dataset_cache'
    cache.mkdir(parents=True,exist_ok=True)
    bundle=pilot.prepare_imagenette(project,cache,None,config['seed'],logger)
    stored=pd.read_csv(run/'dataset_manifest_imagenette.csv').sort_values('global_dataset_index')
    fresh=bundle.manifest.sort_values('global_dataset_index')
    for column in ['global_dataset_index','target_imagenet1000','relative_path']:
        if stored[column].tolist()!=fresh[column].tolist():
            raise RuntimeError('Dataset manifest differs: '+column)
    split=pd.read_csv(run/'calibration_split_manifest.csv')
    indices=np.sort(split.loc[split.split=='calibration','global_dataset_index'].to_numpy(dtype=np.int64))
    assert len(indices)==1963
    digest=config['design_digest']
    identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'fit_parameters').glob('*.json')}
    for alias in config['models']:
        print('Regenerating calibration logits:',alias,flush=True)
        model,mean,std=pilot.load_model(alias,device,project/'model_cache')
        pilot.ensure_calibration_data(model,mean,std,alias,bundle,indices,design,digest,run,device,logger)
        del model
        torch.cuda.empty_cache()
    assert identity=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'fit_parameters').glob('*.json')}
    record={'torch':torch.__version__,'torchvision':pilot.torchvision.__version__,'timm':pilot.timm.__version__,'gpu':torch.cuda.get_device_name(0),'regenerated':True,'fitted_parameters_unchanged':True,'source_config_sha256':hashlib.sha256((run/'config.json').read_bytes()).hexdigest()}
    (project/'REGENERATED_FIT_LOGITS.json').write_text(json.dumps(record,indent=2))
    print('Calibration-logit cache ready; fitted parameters unchanged.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True,help='New writable project; contains datasets/ and runs/.')
    parser.add_argument('--regenerate-fit-logits',action='store_true')
    parser.add_argument('--batch-size',type=int,default=24)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    project=args.project.resolve();project.mkdir(parents=True,exist_ok=True)
    if project==ROOT:raise ValueError('Use a separate project directory.')
    run=stage(project)
    if args.regenerate_fit_logits:regenerate(project,run,args.batch_size,args.workers)

if __name__=='__main__':main()

