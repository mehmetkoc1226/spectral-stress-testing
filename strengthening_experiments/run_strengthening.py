"""Two additive experiments, using the supplied scientific inference modules.

Original fit parameters, data split, model specifications and input transforms
are frozen. New run paths are separate. All evaluation logits are retained.
"""
from pathlib import Path
import argparse,dataclasses,gc,hashlib,json,os,time
import numpy as np
import pandas as pd
from scipy.special import softmax,expit
from scipy.optimize import brentq
from spectral_controls import match_moments

ORIGINAL='spectral_stage2_calibration_pilot_v1__27bb9eee96fb'
MODELS=('resnet50_a1','swin_t_in1k','vitb16_in1k')
VERSION='spectral-strengthening-20261027-v1'

def scalar_temperature(z,y):
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=int);zy=z[np.arange(len(z)),y]
    def g(v):return float(np.mean(np.sum(softmax(v*z,axis=1)*z,axis=1)-zy))
    lo,hi=.05,20.
    inv=lo if g(lo)>=0 else hi if g(hi)<=0 else brentq(g,lo,hi,xtol=1e-12)
    return 1/inv

def save_csv(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    frame.to_csv(tmp,index=False,compression='gzip' if path.name.endswith('.gz') else None);os.replace(tmp,path)

def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def freeze(path,payload):
    text=json.dumps(payload,indent=2,sort_keys=True)
    if path.exists():
        if json.loads(path.read_text())!=payload:raise RuntimeError('Frozen protocol mismatch: '+str(path))
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def main(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('This run requires a GPU runtime; CPU-only execution is not started.')
    import spectral_stage2_core as core
    import spectral_stage2_pilot as pilot
    project=args.project.resolve();original=project/'runs'/ORIGINAL
    config=json.loads((original/'config.json').read_text())
    if config['design_digest']!='27bb9eee96fb' or set(config['models'])!=set(MODELS):raise RuntimeError('Original design mismatch')
    split=pd.read_csv(original/'calibration_split_manifest.csv')
    fit_ids=set(split.loc[split.split=='calibration','global_dataset_index'])
    eval_ids=set(split.loc[split.split=='evaluation','global_dataset_index'])
    assert len(fit_ids)==1963 and len(eval_ids)==1962 and not fit_ids.intersection(eval_ids)
    pilot.seed_everything(config['seed'])
    device=torch.device('cuda')
    cases=[]
    if args.task=='controls':
        cases.append('clean_common')
        for family,prefix in [('hard','hard'),('gaussian','gauss')]:
            for radius in [10,20]:
                for mode,short in [('low_pass','lp'),('high_pass','hp')]:
                    base=f'{prefix}_{short}_r{radius:03d}_linear';cases.append(base)
                    for variant in (['dc','rms'] if short=='hp' else ['rms']):
                        key=f'{prefix}_{short}_r{radius:03d}_{variant}';cases.append(key)
                        pilot.CATALOG[key]=core.Condition(key,family,mode,radius,variant)
        old_apply=pilot.apply_condition
        def controlled(images,condition,indices,seed,dataset):
            if condition.postprocess in ['dc','rms']:
                band=core.apply_spectral_filter(images,dataclasses.replace(condition,postprocess='linear'))
                return match_moments(images,band,condition.postprocess)
            return old_apply(images,condition,indices,seed,dataset)
        pilot.apply_condition=controlled
    else:cases=list(config['imagenette_conditions'])
    protocol=dict(version=VERSION,task=args.task,mode=args.mode,models=list(MODELS),original_run=ORIGINAL,
        original_config_sha256=hashlib.sha256((original/'config.json').read_bytes()).hexdigest(),
        fit_parameters_sha256={m:hashlib.sha256((original/'fit_parameters'/f'{m}.json').read_bytes()).hexdigest() for m in MODELS},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        controls_sha256=hashlib.sha256(Path(__file__).with_name('spectral_controls.py').read_bytes()).hexdigest(),
        imagenette_conditions=cases,imagenet_a_conditions=config['imagenet_a_conditions'] if args.task=='calibration' else [],
        batch_size=args.batch_size,mixed_precision=config['mixed_precision'],seed=config['seed'],
        curve_grid=[0,.25,.5,.75,1] if args.curves else [],
        selection='No ImageNet-A selection; pooled scalar fitted on original calibration views only; curve grid reported in full',
        logits_retained=True)
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:12]
    run=project/'runs'/f'spectral_strengthening_{args.task}_{args.mode}__{digest}'
    assert run.resolve()!=original.resolve() and ORIGINAL not in run.name
    freeze(run/'PROTOCOL.json',protocol)
    logger=pilot.setup_logger(run/'run.log')
    print('Output:',run,flush=True)
    logger.info('GPU %s',torch.cuda.get_device_name(0))
    logger.info('Protocol locked before inference; no original run files are replaced')
    (run/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,
        torchvision=pilot.torchvision.__version__,timm=pilot.timm.__version__,gpu=torch.cuda.get_device_name(0)),indent=2))
    # GPU-side implementation parity with the independently checked NumPy formula.
    rng=np.random.default_rng(91);x=rng.uniform(.1,.9,(3,3,32,32)).astype(np.float32);b=rng.normal(0,.1,x.shape).astype(np.float32)
    for variant in ['dc','rms']:
        a=match_moments(torch.tensor(x,device=device),torch.tensor(b,device=device),variant).cpu().numpy()
        assert np.allclose(a,match_moments(x,b,variant),rtol=2e-5,atol=2e-6)
    core_checks=core.self_test();(run/'scientific_self_tests.json').write_text(json.dumps(core_checks,indent=2))
    local=Path('/content/spectral_strengthening_data');local.mkdir(parents=True,exist_ok=True)
    bundles={'imagenette':pilot.prepare_imagenette(project,local,None,config['seed'],logger)}
    if args.task=='calibration':bundles['imagenet_a']=pilot.prepare_imagenet_a(project,local,None,config['seed'],logger)
    for ds,bundle in bundles.items():
        old=pd.read_csv(original/f'dataset_manifest_{ds}.csv').sort_values('global_dataset_index')
        fresh=bundle.manifest.sort_values('global_dataset_index')
        assert old.global_dataset_index.tolist()==fresh.global_dataset_index.tolist()
        assert old.target_imagenet1000.tolist()==fresh.target_imagenet1000.tolist()
        assert old.relative_path.tolist()==fresh.relative_path.tolist()
    design=dataclasses.replace(pilot.build_design(),run_name=run.name,code_version=VERSION,profile='paper',
        models=MODELS,batch_size=args.batch_size,chunk_size=128,num_workers=args.workers,
        mixed_precision=config['mixed_precision'],seed=config['seed'])
    summary=[];benchmark=[];agreements=[];probe_token=str(time.time_ns())
    for model_alias in MODELS:
        model,mean,std=pilot.load_model(model_alias,device,project/'model_cache')
        fit=json.loads((original/'fit_parameters'/f'{model_alias}.json').read_text())
        pooled={}
        if args.task=='calibration':
            cal=[load_npz(original/'calibration_logits'/model_alias/(c+'.npz')) for c in config['calibration_conditions']]
            for a in cal:assert set(a['global_indices'])==fit_ids
            zs=np.concatenate([a['logits'] for a in cal]);ys=np.concatenate([a['targets1000'] for a in cal])
            for protocol_name in ['imagenet1000','restricted10']:
                z,y,_=core.protocol_view(zs,ys,bundles['imagenette'].restricted_indices,protocol_name)
                pooled[protocol_name]=scalar_temperature(z,y)
            pilot.atomic_write_json(run/'pooled_fit'/f'{model_alias}.json',dict(temperatures=pooled,n_unique_images=1963,n_views=5889))
        try:
            for dataset,bundle in bundles.items():
                indices=bundle.active_indices if args.task=='controls' or dataset=='imagenet_a' else np.array(sorted(eval_ids),dtype=np.int64)
                conditions=cases if dataset=='imagenette' else config['imagenet_a_conditions']
                if args.mode=='benchmark':
                    indices=indices[:128]
                    conditions=['clean_common',cases[-1]] if args.task=='controls' else ['clean_common','hard_hp_r020_linear']
                for condition in conditions:
                    audit_path=original/'audit_logits'/dataset/model_alias/(condition+'.npz')
                    if args.mode=='benchmark' and args.task=='calibration' and audit_path.exists():
                        indices=np.sort(load_npz(audit_path)['global_indices'][:128]).astype(np.int64)
                    raw=run/'raw_logits'/dataset/model_alias/condition
                    if args.mode=='benchmark':raw=run/'timing_probes'/probe_token/dataset/model_alias/condition
                    t0=time.perf_counter()
                    arrays=pilot.collect_raw_arrays(model,mean,std,bundle,indices,condition,raw,design,digest,device,logger)
                    elapsed=time.perf_counter()-t0
                    benchmark.append(dict(model=model_alias,dataset=dataset,condition=condition,n=len(indices),seconds=elapsed))
                    assert np.array_equal(arrays['global_indices'],indices)
                    if dataset=='imagenette' and args.task=='calibration':assert not set(indices).intersection(fit_ids)
                    if args.task=='calibration' and audit_path.exists():
                        oldaudit=load_npz(audit_path);common,ni,oi=np.intersect1d(arrays['global_indices'],oldaudit['global_indices'],return_indices=True)
                        A=arrays['logits'][ni].astype(np.float64);O=oldaudit['logits'][oi].astype(np.float64)
                        agreements.append(dict(dataset=dataset,model=model_alias,condition=condition,n_audit=len(common),
                            prediction_mismatches=int(np.sum(A.argmax(1)!=O.argmax(1))),max_abs_logit_difference=float(np.max(np.abs(A-O)))))
                        save_csv(run/'REPLAY_AGREEMENT.csv',pd.DataFrame(agreements))
                    for output_protocol in ['imagenet1000',f'restricted{len(bundle.restricted_indices)}']:
                        z,y,classes=core.protocol_view(arrays['logits'],arrays['targets1000'],bundle.restricted_indices,output_protocol)
                        temps={'raw':1.}
                        if args.task=='calibration':
                            ps=fit['protocols']['imagenet1000' if output_protocol=='imagenet1000' else 'restricted10']
                            clean=float(ps['clean_scalar_temperature']);temps.update(clean_ts=clean,pooled_ts=pooled['imagenet1000' if output_protocol=='imagenet1000' else 'restricted10'])
                            for key,method in [('frequency_conditioned','fcts'),('confidence_conditioned','confidence_ts')]:
                                features=arrays['descriptors'] if method=='fcts' else core.confidence_features(z)
                                standard=fit['descriptor_standardization'] if method=='fcts' else ps['confidence_standardization']
                                b=np.asarray(ps[key]['parameters']);X=core.apply_standardizer(features,standard);index=b[0]+X@b[1:]
                                temps[method]=.05+19.95*expit(index)
                                if args.curves:
                                    clean_index=np.log((clean-.05)/(20-clean))
                                    for lam in [.25,.5,.75]:temps[f'{method}_lambda{int(100*lam):03d}']=.05+19.95*expit(clean_index+lam*(index-clean_index))
                        frame=pd.DataFrame(dict(dataset=dataset,model=model_alias,condition=condition,protocol=output_protocol,
                            global_dataset_index=arrays['global_indices'],target_imagenet1000=arrays['targets1000']))
                        expected_pred=classes[np.argmax(z,axis=1)]
                        for method,T in temps.items():
                            predictions=core.method_outputs(z,y,classes,T)
                            assert np.array_equal(predictions['prediction'],expected_pred)
                            metrics=core.metrics_from_outputs(predictions)
                            summary.append(dict(dataset=dataset,model=model_alias,condition=condition,protocol=output_protocol,method=method,n=len(y),**metrics))
                            for col,value in predictions.items():frame[method+'_'+col]=value
                        save_csv(run/'compact_predictions'/dataset/model_alias/f'{condition}__{output_protocol}.csv.gz',frame)
                    save_csv(run/'summary.csv',pd.DataFrame(summary))
                    save_csv(run/'timing.csv',pd.DataFrame(benchmark))
                    logger.info('DONE %s %s %s n=%d %.1fs',model_alias,dataset,condition,len(indices),elapsed)
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    if args.mode=='benchmark':
        df=pd.DataFrame(benchmark);estimates={}
        for model in MODELS:
            q=df[df.model==model];seconds=0
            for ds in bundles:
                rate=(q[q.dataset==ds].seconds/q[q.dataset==ds].n).max()
                images=3925*len(cases) if args.task=='controls' else (1962*15 if ds=='imagenette' else 7500*11)
                seconds+=rate*images
            estimates[model]=float(seconds/60)
        pilot.atomic_write_json(run/'TIMING_ESTIMATE.json',dict(inference_minutes_by_model=estimates,total_inference_minutes=sum(estimates.values()),
            note='Fresh 128-image inference probes including preprocessing/shard write. Excludes data staging, model download, scalar fitting, probability scoring and bootstrap. Hardware dependent. Cached probes are not reused for timing.'))
        print('Estimated full inference minutes (fitting/scoring/analysis additional):',sum(estimates.values()),flush=True)
    else:
        expected=3*((21*3925) if args.task=='controls' else (15*1962+11*7500))
        observed=int(pd.DataFrame(benchmark).n.sum());assert observed==expected,(observed,expected)
        pilot.atomic_write_json(run/'RUN_COMPLETE.json',dict(task=args.task,rows=len(summary),image_forwards=observed,logits_retained=True))
        from analyse_strengthening import analyse
        analyse(run)
    return run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--task',choices=['controls','calibration'],required=True)
    p.add_argument('--mode',choices=['benchmark','full'],default='benchmark');p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--curves',action='store_true');a=p.parse_args();main(a)
