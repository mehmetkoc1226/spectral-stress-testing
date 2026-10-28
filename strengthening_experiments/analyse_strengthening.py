"""Paired image bootstrap for the two additive frozen-checkpoint experiments."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

FAMILIES=['unseen_radius','unseen_filter','natural_corruption','unseen_dataset']

def family(ds,c):
    if ds=='imagenet_a':return 'unseen_dataset'
    if c=='clean_common':return 'heldout_clean'
    if c.startswith('gauss'):return 'unseen_filter'
    if c.startswith(('blur','noise')):return 'natural_corruption'
    return 'seen_spectral' if '_r020_' in c else 'unseen_radius'

def ece_basis(c,a):
    ix=np.searchsorted(np.linspace(0,1,16)[1:],c,side='left').clip(0,14)
    basis=np.zeros((len(c),15));basis[np.arange(len(c)),ix]=a-c
    return basis

def interval(point,draws):
    low,high=np.quantile(draws,[.025,.975]);se=np.std(draws,ddof=1)
    return dict(estimate=float(point),ci_low=float(low),ci_high=float(high),bootstrap_se=float(se),
        p_value=float(2*norm.sf(abs(point)/se)) if se else float(point==0))

def holm(p):
    p=np.asarray(p);o=np.argsort(p);r=np.empty(len(p));r[o]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[o]));return r

def load_frames(run):
    frames={}
    for path in sorted((run/'compact_predictions').rglob('*.csv.gz')):
        d=pd.read_csv(path).sort_values('global_dataset_index')
        assert not d.global_dataset_index.duplicated().any()
        assert np.isfinite(d.select_dtypes(include=[np.number]).to_numpy()).all()
        key=tuple(str(d[c].iloc[0]) for c in ['dataset','model','condition','protocol'])
        assert key not in frames;frames[key]=d
    return frames

def control_analysis(frames,out,B,seed):
    rng=np.random.default_rng(seed);ids=None;W=None;rows=[]
    for (ds,model,c,protocol),hp in frames.items():
        if not c.endswith('_hp_r010_linear') and not c.endswith('_hp_r020_linear'):continue
        lpkey=c.replace('_hp_','_lp_');hpdc=c.replace('_linear','_dc');hprms=c.replace('_linear','_rms');lprms=lpkey.replace('_linear','_rms')
        pairs=[('signed_hp_minus_lp',c,lpkey),('dc_restore_hp',hpdc,c),('rms_hp_minus_lp',hprms,lprms),('rms_after_dc_hp',hprms,hpdc)]
        for contrast,a,b in pairs:
            A=frames[(ds,model,a,protocol)];D=frames[(ds,model,b,protocol)]
            if ids is None:
                ids=A.global_dataset_index.to_numpy();W=rng.multinomial(len(A),np.full(len(A),1/len(A)),size=B)
            assert np.array_equal(ids,A.global_dataset_index) and np.array_equal(ids,D.global_dataset_index)
            delta=A.raw_correct.to_numpy(dtype=float)-D.raw_correct.to_numpy(dtype=float)
            rows.append(dict(dataset=ds,model=model,condition_a=a,condition_b=b,protocol=protocol,contrast=contrast,
                metric='accuracy',n=len(A),**interval(delta.mean(),W@delta/len(delta))))
    df=pd.DataFrame(rows);df['p_holm']=np.nan
    for _,g in df.groupby(['contrast','metric']):df.loc[g.index,'p_holm']=holm(g.p_value.to_numpy())
    df.to_csv(out/'CONTROL_PAIRED_ACCURACY.csv',index=False)
    return df

def calibration_analysis(frames,out,B,seed):
    rng=np.random.default_rng(seed);W={};ids={};cells=[];point=[];draw=[];methods=None
    for (ds,model,c,protocol),d in frames.items():
        if protocol!='imagenet1000':continue
        now=[col[:-4] for col in d.columns if col.endswith('_nll')]
        if methods is None:methods=now
        assert methods==now
        n=len(d)
        if ds not in W:
            ids[ds]=d.global_dataset_index.to_numpy();W[ds]=rng.multinomial(n,np.full(n,1/n),size=B).astype(float)
        assert np.array_equal(ids[ds],d.global_dataset_index)
        blocks=[];pt=[]
        for m in methods:
            residual=ece_basis(d[m+'_confidence'].to_numpy(),d[m+'_correct'].to_numpy(dtype=float))
            blocks.append(np.column_stack([d[m+'_nll'],d[m+'_brier'],residual]))
            pt.append([d[m+'_nll'].mean(),np.abs(residual.sum(0)).sum()/n,d[m+'_brier'].mean()])
        samples=W[ds]@np.column_stack(blocks)/n
        reps=np.stack([np.column_stack([samples[:,j*17],np.abs(samples[:,j*17+2:j*17+17]).sum(1),samples[:,j*17+1]]) for j in range(len(methods))],axis=1)
        cells.append(dict(dataset=ds,model=model,condition=c,family=family(ds,c)));point.append(np.array(pt));draw.append(reps)
    models=sorted(set(c['model'] for c in cells));rows=[];scores=[]
    pairs=[('fcts','clean_ts'),('confidence_ts','clean_ts'),('pooled_ts','clean_ts'),('fcts','confidence_ts'),('fcts','pooled_ts'),('confidence_ts','pooled_ts')]
    # Predefined curve methods are all reported; no winner selection on evaluation data.
    pairs += [(m,'clean_ts') for m in methods if '_lambda' in m]
    for scope in ['all']+FAMILIES+['heldout_clean']:
        fpoint=[];fdraw=[]
        for model in models:
            for fam in FAMILIES if scope=='all' else [scope]:
                ix=[i for i,c in enumerate(cells) if c['model']==model and c['family']==fam]
                if not ix:raise ValueError('Incomplete model/family grid')
                fpoint.append(np.mean([point[i] for i in ix],axis=0));fdraw.append(np.mean([draw[i] for i in ix],axis=0))
        pt=np.mean(fpoint,axis=0);rep=np.mean(fdraw,axis=0)
        for j,m in enumerate(methods):scores.append(dict(scope=scope,method=m,nll=pt[j,0],ece15=pt[j,1],brier=pt[j,2]))
        for a,b in pairs:
            ia,ib=methods.index(a),methods.index(b)
            for k,metric in enumerate(['nll','ece15','brier']):
                rows.append(dict(scope=scope,method_a=a,method_b=b,metric=metric,**interval(pt[ia,k]-pt[ib,k],rep[:,ia,k]-rep[:,ib,k])))
    result=pd.DataFrame(rows);result.to_csv(out/'CALIBRATION_PAIRED_CONTRASTS.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'CALIBRATION_BALANCED_SCORES.csv',index=False)
    return result

def analyse(run,B=1000,seed=20261027):
    run=Path(run);protocol=json.loads((run/'PROTOCOL.json').read_text());complete=json.loads((run/'RUN_COMPLETE.json').read_text())
    assert protocol['mode']=='full' and complete['task']==protocol['task']
    out=run/'analysis';out.mkdir(exist_ok=True);frames=load_frames(run)
    with threadpool_limits(limits=2):
        result=control_analysis(frames,out,B,seed) if protocol['task']=='controls' else calibration_analysis(frames,out,B,seed)
    (out/'ANALYSIS_PROTOCOL.json').write_text(json.dumps(dict(B=B,seed=seed,scope='Fixed checkpoints/calibrators; shared image weights within dataset; independent datasets',
        curves='All predefined grid points reported; no selection from ImageNet-A or held-out test performance'),indent=2))
    print(result.head(12).round(6).to_string(index=False),flush=True)
    print('Analysis saved:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();analyse(a.run)
