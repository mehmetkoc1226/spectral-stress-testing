"""CPU numerical checks; this file does not execute a backbone."""
from pathlib import Path
import tempfile,json,hashlib
import numpy as np
import pandas as pd
from spectral_controls import self_check
from run_strengthening import scalar_temperature,freeze
from analyse_strengthening import ece_basis,load_frames,calibration_analysis,control_analysis

def checks():
    passed=self_check()['checks']
    z=np.tile([2.,0.],(10,1));y=np.r_[np.zeros(9,dtype=int),1]
    assert np.isclose(scalar_temperature(z,y),2/np.log(9),rtol=1e-9)
    passed.append('Pooled scalar optimum matches analytic binary solution')
    c=np.array([0,1/15,2/15,.5,1.]);a=np.array([0,1,0,1,0.]);w=np.array([1,2,0,3,4])
    weighted=np.abs(w@ece_basis(c,a)).sum()/w.sum()
    literal=np.abs(ece_basis(np.repeat(c,w),np.repeat(a,w)).sum(0)).sum()/w.sum()
    assert np.isclose(weighted,literal);passed.append('Right-closed weighted ECE matches literal resampling at boundaries')
    models=['resnet50_a1','swin_t_in1k','vitb16_in1k'];frames={};n=24
    for ds in ['imagenette','imagenet_a']:
        cases=['clean_common','hard_lp_r010_linear','hard_hp_r010_linear','hard_lp_r020_linear','hard_hp_r020_linear',
            'gauss_lp_r010_linear','gauss_hp_r010_linear','gauss_lp_r020_linear','gauss_hp_r020_linear','blur_sigma4','noise_std020']
        if ds=='imagenette':cases+=['hard_lp_r040_linear','hard_hp_r040_linear','blur_sigma2','noise_std010']
        for mi,model in enumerate(models):
            for ci,condition in enumerate(cases):
                for protocol in ['imagenet1000','restricted10' if ds=='imagenette' else 'restricted200']:
                    d=pd.DataFrame(dict(dataset=ds,model=model,condition=condition,protocol=protocol,
                        global_dataset_index=np.arange(n),target_imagenet1000=np.arange(n)%2))
                    for j,method in enumerate(['raw','clean_ts','pooled_ts','fcts','confidence_ts']):
                        c0=np.full(n,.65+.025*j);correct=(np.arange(n)%3!=0)
                        d[method+'_confidence']=c0;d[method+'_correct']=correct
                        d[method+'_nll']=np.full(n,2+ci*.01+mi*.02-j*.1)
                        d[method+'_brier']=np.full(n,.5-j*.02)
                    frames[(ds,model,condition,protocol)]=d
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);r=calibration_analysis(frames,out,100,20261027)
        row=r[(r.scope=='all')&(r.method_a=='fcts')&(r.method_b=='clean_ts')&(r.metric=='nll')].iloc[0]
        assert np.isclose(row.estimate,-.2) and np.isclose(row.ci_low,-.2) and np.isclose(row.ci_high,-.2)
        passed.append('Family/model balancing and shared-image calibration CI recover constant paired effect')
        p=out/'frozen.json';freeze(p,{'a':1});freeze(p,{'a':1})
        try:freeze(p,{'a':2})
        except RuntimeError:pass
        else:raise AssertionError('Protocol overwrite accepted')
        passed.append('Resume accepts the same protocol and rejects a changed protocol')
        controls={}
        for model in models:
            for protocol in ['imagenet1000','restricted10']:
                for prefix in ['hard','gauss']:
                    for radius in [10,20]:
                        for short,variant in [('lp','linear'),('hp','linear'),('lp','rms'),('hp','rms'),('hp','dc')]:
                            key=f'{prefix}_{short}_r{radius:03d}_{variant}'
                            d=pd.DataFrame(dict(global_dataset_index=np.arange(n),raw_correct=np.arange(n)%2==0))
                            controls[('imagenette',model,key,protocol)]=d
        r=control_analysis(controls,out,100,20261027)
        assert len(r)==96 and np.allclose(r.estimate,0) and np.allclose(r.p_holm,1)
        passed.append('All 96 planned control contrasts paired correctly; identical records yield zero effects')
    upstream=json.loads(Path(__file__).with_name('UPSTREAM_MODULES.json').read_text())
    for name,h in upstream['sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    passed.append('Extracted upstream scientific modules match recorded byte hashes')
    return dict(status='passed',n_checks=len(passed),checks=passed,gpu_inference_executed=False,new_empirical_experiment_results=False)

if __name__=='__main__':print(json.dumps(checks(),indent=2))
