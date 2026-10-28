"""Full-cohort record audit and post-hoc pooled-TS check on saved logit panels.

No backbone execution. The pooled scalar uses all 1,963 fitting images in
each of the three original fitting conditions. Its evaluation is restricted
to 256 saved logits per model/condition, which differ between panels.
"""
from pathlib import Path
import argparse, json, zipfile
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax, expit
from scipy.optimize import brentq
from scipy.stats import norm
from threadpoolctl import threadpool_limits

METHODS=['raw','clean_ts','confidence_ts','fcts','oracle_ts']
PRIMARY=['unseen_radius','unseen_filter','natural_corruption','unseen_dataset']
MODELS=['resnet50_a1','swin_t_in1k','vitb16_in1k']
EPS=1e-12

def family(dataset,condition):
    if dataset=='imagenet_a':return 'unseen_dataset'
    if condition=='clean_common':return 'heldout_clean'
    if condition.startswith('gauss'):return 'unseen_filter'
    if condition.startswith(('blur','noise')):return 'natural_corruption'
    if '_r020_' in condition:return 'seen_spectral'
    return 'unseen_radius'

def ece_basis(confidence,correct):
    # First bin [0,1/15], all others (b/15,(b+1)/15].
    bins=np.searchsorted(np.linspace(0,1,16)[1:],confidence,side='left').clip(0,14)
    a=np.zeros((len(confidence),15),dtype=np.float64)
    a[np.arange(len(a)),bins]=correct-confidence
    return a

def ci_row(point,rep):
    lo,hi=np.quantile(rep,[.025,.975]);se=np.std(rep,ddof=1)
    p=2*norm.sf(abs(point)/se) if se>0 else (1.0 if point==0 else 0.0)
    return dict(estimate=float(point),ci_low=float(lo),ci_high=float(hi),bootstrap_se=float(se),p_value=float(p))

def holm(values):
    p=np.asarray(values);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return out

def compact_audit(data,report,out,B,seed):
    summary=pd.read_csv(report/'summary.csv')
    z=zipfile.ZipFile(data/'stage2_compact.zip')
    rng=np.random.default_rng(seed)
    weights={};ids={};points=[];cells=[];draws=[];full={};floor={m:0 for m in METHODS}
    mismatches=0;rowcount=0;maxdiff={k:0. for k in ['accuracy','nll','ece15','brier']}
    for k,name in enumerate(n for n in z.namelist() if n.endswith('.csv.gz')):
        df=pd.read_csv(z.open(name),compression='gzip').sort_values('global_dataset_index')
        assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
        dataset,model,condition,protocol=[str(df[c].iloc[0]) for c in ['dataset','model','condition','protocol']]
        key=(dataset,model,condition,protocol);full[key]=df
        assert not df.global_dataset_index.duplicated().any()
        if dataset not in ids:
            ids[dataset]=df.global_dataset_index.to_numpy()
            weights[dataset]=rng.multinomial(len(df),np.full(len(df),1/len(df)),size=B).astype(np.float64)
        assert np.array_equal(ids[dataset],df.global_dataset_index.to_numpy())
        blocks=[]; n=len(df); correct=df.raw_correct.to_numpy(dtype=float)
        rowcount+=n
        for m in METHODS:
            mismatches+=int(np.sum(df[m+'_prediction'].to_numpy()!=df.raw_prediction.to_numpy()))
            assert np.array_equal(df[m+'_correct'].to_numpy(),df.raw_correct.to_numpy())
            assert np.array_equal(df[m+'_prediction'].to_numpy()==df.target_imagenet1000.to_numpy(),df[m+'_correct'].to_numpy())
            assert df[m+'_confidence'].between(0,1).all() and df[m+'_true_probability'].between(0,1).all()
            floor[m]+=int(np.sum(df[m+'_true_probability'].to_numpy()<=EPS*1.001))
            conf=df[m+'_confidence'].to_numpy(dtype=float)
            basis=ece_basis(conf,correct)
            metrics=dict(accuracy=float(correct.mean()),ece15=float(np.abs(basis.sum(axis=0)).sum()/n),
                nll=float(df[m+'_nll'].mean()),brier=float(df[m+'_brier'].mean()))
            points.append(dict(dataset=dataset,model=model,condition=condition,protocol=protocol,method=m,**metrics))
            old=summary[(summary.dataset==dataset)&(summary.model==model)&(summary.condition==condition)&(summary.protocol==protocol)&(summary.method==m)].iloc[0]
            for metric in maxdiff:maxdiff[metric]=max(maxdiff[metric],abs(metrics[metric]-float(old[metric])))
            if m in ['fcts','clean_ts','confidence_ts']:
                blocks.append(np.column_stack([df[m+'_nll'],df[m+'_brier'],basis]))
        reps=weights[dataset]@np.column_stack(blocks)/n
        # block order clean_ts, confidence_ts, fcts, because METHODS order.
        vals=np.stack([np.column_stack([reps[:,i*17],np.abs(reps[:,i*17+2:i*17+17]).sum(axis=1),reps[:,i*17+1]]) for i in range(3)],axis=1)
        pt=np.array([[next(p for p in points[-5:] if p['method']==m)[metric] for metric in ['nll','ece15','brier']] for m in ['clean_ts','confidence_ts','fcts']])
        cells.append(dict(dataset=dataset,model=model,condition=condition,protocol=protocol,evaluation_family=family(dataset,condition)))
        draws.append((pt,vals))
        if (k+1)%26==0:print('Audited compact files',k+1,flush=True)
    assert rowcount==671580 and mismatches==0,(rowcount,mismatches)
    assert max(maxdiff.values())<2e-6,maxdiff
    pd.DataFrame(points).to_csv(out/'calibration_recomputed_metrics.csv',index=False)
    contrast=[]
    for c,(pt,vals) in zip(cells,draws):
        for target,j,base in [(2,0,'clean_ts'),(2,1,'confidence_ts'),(1,0,'clean_ts')]:
            for mi,metric in enumerate(['nll','ece15','brier']):
                contrast.append(dict(**c,method='fcts' if target==2 else 'confidence_ts',baseline=base,metric=metric,**ci_row(pt[target,mi]-pt[j,mi],vals[:,target,mi]-vals[:,j,mi])))
    contrasts=pd.DataFrame(contrast)
    contrasts['p_holm']=np.nan
    for _,g in contrasts.groupby(['method','baseline','evaluation_family','metric']):
        contrasts.loc[g.index,'p_holm']=holm(g.p_value.to_numpy())
    contrasts.to_csv(out/'calibration_cell_contrasts.csv',index=False)
    aggregate=[]
    # Equal weight per family, then per anchor. Model/family subaggregates retain this order.
    for target,base_i,base in [(2,0,'clean_ts'),(2,1,'confidence_ts'),(1,0,'clean_ts')]:
        for model_sel in ['all']+MODELS:
            for family_sel in ['all']+PRIMARY+['heldout_clean','seen_spectral']:
                group_point=[];group_draw=[]
                for model in MODELS if model_sel=='all' else [model_sel]:
                    fam_point=[];fam_draw=[]
                    for fam in PRIMARY if family_sel=='all' else [family_sel]:
                        ix=[i for i,c in enumerate(cells) if c['model']==model and c['protocol']=='imagenet1000' and c['evaluation_family']==fam]
                        assert ix
                        fam_point.append(np.mean([draws[i][0][target]-draws[i][0][base_i] for i in ix],axis=0))
                        fam_draw.append(np.mean([draws[i][1][:,target]-draws[i][1][:,base_i] for i in ix],axis=0))
                    group_point.append(np.mean(fam_point,axis=0));group_draw.append(np.mean(fam_draw,axis=0))
                pt=np.mean(group_point,axis=0);rep=np.mean(group_draw,axis=0)
                for mi,metric in enumerate(['nll','ece15','brier']):
                    aggregate.append(dict(model=model_sel,evaluation_family=family_sel,method='fcts' if target==2 else 'confidence_ts',baseline=base,metric=metric,**ci_row(pt[mi],rep[:,mi])))
    agg=pd.DataFrame(aggregate);agg.to_csv(out/'calibration_balanced_contrasts.csv',index=False)
    rp=[];rb=[];condition_values=[]
    for model in MODELS:
        for fam in PRIMARY:
            ix=[i for i,c in enumerate(cells) if c['model']==model and c['protocol']!='imagenet1000' and c['evaluation_family']==fam]
            rp.append(np.mean([draws[i][0][2,0]-draws[i][0][1,0] for i in ix]))
            rb.append(np.mean([draws[i][1][:,2,0]-draws[i][1][:,1,0] for i in ix],axis=0))
            condition_values.extend(draws[i][0][2,0]-draws[i][0][1,0] for i in ix)
    restricted=ci_row(np.mean(rp),np.mean(rb,axis=0))
    restricted.update(condition_weighted_estimate=float(np.mean(condition_values)),B=B,seed=seed)
    (out/'restricted_balanced_nll.json').write_text(json.dumps(restricted,indent=2))
    audit=dict(files=len(cells),rows=rowcount,shared_index_alignment=True,top1_mismatches=mismatches,
        nonfinite_records=0,correctness_mismatches=0,duplicate_index_files=0,max_summary_differences=maxdiff,
        probability_floor=EPS,floor_hits=floor,bootstrap_replicates=B,bootstrap_seed=seed,
        bootstrap_scope='One shared image-multiplicity matrix per dataset across all conditions, models, methods and output tasks; datasets independent; fitted calibrators fixed',
        ece_convention='[0,1/15], (1/15,2/15], ..., (14/15,1]',
        supplied_code_version='v3',archived_run_code_version='v4; exact v4 notebook not supplied')
    (out/'calibration_audit.json').write_text(json.dumps(audit,indent=2))
    print('Balanced compact contrasts:',agg[(agg.model=='all')&(agg.evaluation_family=='all')].to_string(index=False),flush=True)
    return full

def view(logits,targets,restricted):
    z=np.asarray(logits,dtype=np.float64)
    if restricted is None:return z,np.asarray(targets,dtype=int),np.arange(1000)
    classes=np.asarray(restricted,dtype=int)
    return z[:,classes],np.searchsorted(classes,targets),classes

def fit_scalar(z,y):
    zy=z[np.arange(len(z)),y]
    def grad(v):return float(np.mean(np.sum(softmax(v*z,axis=1)*z,axis=1)-zy))
    lo,hi=.05,20.
    v=lo if grad(lo)>=0 else (hi if grad(hi)<=0 else brentq(grad,lo,hi,xtol=1e-12))
    return 1/v

def outputs(z,y,classes,t):
    t=np.broadcast_to(t,len(z));p=softmax(z/t[:,None],axis=1);pred=p.argmax(axis=1)
    true=np.maximum(p[np.arange(len(p)),y],EPS)
    return dict(prediction=classes[pred],confidence=p.max(axis=1),correct=pred==y,nll=-np.log(true),
        brier=np.sum(p*p,axis=1)-2*true+1,temperature=t)

def conf_features(z):
    p=softmax(z,axis=1);top=np.partition(z,-2,axis=1)[:,-2:]
    return np.column_stack([top[:,1]-top[:,0],-np.sum(p*np.log(np.maximum(p,EPS)),axis=1)/np.log(z.shape[1]),p.max(axis=1)])

def pooled_panels(data,report,out,full):
    fitrows=[];panelrows=[];agreement=[];stationarity=[]
    for model in MODELS:
        param=json.loads((report/'fit_parameters'/f'{model}.json').read_text())
        fits=[np.load(data/'calibration_logits'/model/(c+'.npz')) for c in ['clean_common','hard_lp_r020_linear','hard_hp_r020_linear']]
        split=pd.read_csv(report/'calibration_split_manifest.csv')
        fit_ids=set(split.loc[split.split=='calibration','global_dataset_index'])
        eval_ids=set(split.loc[split.split=='evaluation','global_dataset_index'])
        assert len(fit_ids)==1963 and len(eval_ids)==1962 and not fit_ids.intersection(eval_ids)
        for f in fits:assert set(f['global_indices'])==fit_ids
        for protocol in ['imagenet1000','restricted10']:
            restricted=None if protocol=='imagenet1000' else fits[0]['restricted_indices']
            zs,ys=[],[]
            for f in fits:
                z,y,_=view(f['logits'],f['targets1000'],restricted);zs.append(z);ys.append(y)
            zall,yall=np.concatenate(zs),np.concatenate(ys)
            pooled=fit_scalar(zall,yall);clean=fit_scalar(zs[0],ys[0]);ps=param['protocols'][protocol]
            fitrows.append(dict(model=model,protocol=protocol,n_unique_fit_images=len(ys[0]),n_fit_views=len(yall),
                pooled_temperature=pooled,refitted_clean_temperature=clean,archived_clean_temperature=ps['clean_scalar_temperature']))
            # Saved coefficients verify penalty scope and agreement with cached fitting logits.
            for kind in ['frequency_conditioned','confidence_conditioned']:
                standard=param['descriptor_standardization'] if kind=='frequency_conditioned' else ps['confidence_standardization']
                features=np.concatenate([f['descriptors'] for f in fits]) if kind=='frequency_conditioned' else conf_features(zall)
                X=(features-np.array(standard['mean']))/np.array(standard['std'])
                b=np.array(ps[kind]['parameters']);sig=expit(b[0]+X@b[1:]);T=.05+19.95*sig
                probs=softmax(zall/T[:,None],axis=1)
                gT=(zall[np.arange(len(zall)),yall]-np.sum(probs*zall,axis=1))/(T*T)
                gindex=gT*19.95*sig*(1-sig)
                g=np.r_[gindex.mean(),X.T@gindex/len(X)]+np.r_[0,.002*b[1:]]
                nll=np.mean(logsumexp(zall/T[:,None],axis=1)-zall[np.arange(len(zall)),yall]/T)
                stationarity.append(dict(model=model,protocol=protocol,method=kind,nll=float(nll),
                    archived_nll=ps[kind]['fitted_nll'],gradient_norm_slopes_only=float(np.linalg.norm(g)),
                    gradient_norm_if_intercept_penalized=float(np.linalg.norm(g+np.r_[.002*b[0],0,0,0]))))
            for path in sorted((data/'audit_logits').glob(f'*/{model}/*.npz')):
                ds=path.parent.parent.name;c=path.stem;a=np.load(path)
                if ds=='imagenette':assert set(a['global_indices']).issubset(eval_ids)
                restrict=None if protocol=='imagenet1000' else a['restricted_indices']
                z,y,classes=view(a['logits'],a['targets1000'],restrict)
                actual_protocol=protocol if ds=='imagenette' or protocol=='imagenet1000' else 'restricted200'
                tmap={'clean_ts':ps['clean_scalar_temperature'],'pooled_ts':pooled}
                for kind,m in [('frequency_conditioned','fcts'),('confidence_conditioned','confidence_ts')]:
                    standard=param['descriptor_standardization'] if m=='fcts' else ps['confidence_standardization']
                    feats=a['descriptors'] if m=='fcts' else conf_features(z)
                    X=(feats-np.array(standard['mean']))/np.array(standard['std']);b=np.array(ps[kind]['parameters'])
                    tmap[m]=.05+19.95*expit(b[0]+X@b[1:])
                for method,temp in tmap.items():
                    pred=outputs(z,y,classes,temp);basis=ece_basis(pred['confidence'],pred['correct'])
                    panelrows.append(dict(dataset=ds,model=model,condition=c,protocol=actual_protocol,method=method,n=len(z),
                        evaluation_family=family(ds,c),nll=float(pred['nll'].mean()),brier=float(pred['brier'].mean()),
                        ece15=float(np.abs(basis.sum(axis=0)).sum()/len(z)),accuracy=float(pred['correct'].mean())))
                    if method!='pooled_ts' and full is not None:
                        original=full[(ds,model,c,actual_protocol)].set_index('global_dataset_index').loc[a['global_indices']]
                        agreement.append(dict(dataset=ds,model=model,condition=c,protocol=actual_protocol,method=method,
                            max_abs_nll_difference=float(np.max(np.abs(original[method+'_nll'].to_numpy()-pred['nll']))),
                            prediction_mismatches=int(np.sum(original[method+'_prediction'].to_numpy()!=pred['prediction']))))
    pd.DataFrame(fitrows).to_csv(out/'pooled_ts_fit.csv',index=False)
    pd.DataFrame(stationarity).to_csv(out/'calibrator_penalty_verification.csv',index=False)
    panels=pd.DataFrame(panelrows);panels.to_csv(out/'pooled_ts_audit_panel_metrics.csv',index=False)
    pd.DataFrame(agreement).to_csv(out/'audit_logit_quantization_agreement.csv',index=False)
    native=panels[(panels.protocol=='imagenet1000')&panels.evaluation_family.isin(PRIMARY)]
    ag=native.groupby(['method','model','evaluation_family'])[['nll','ece15','brier']].mean().groupby(['method','model']).mean().groupby('method').mean()
    ag.to_csv(out/'pooled_ts_audit_balanced.csv')
    (out/'pooled_ts_audit_scope.json').write_text(json.dumps(dict(fit_unique_images=1963,fit_conditions=3,fit_views_per_model_protocol=5889,
        evaluation_images_per_panel=256,panels=78,full_evaluation_logits_available=False,
        important='Audit subsets selected separately per model and condition; comparisons within each panel use identical images. No shared-image CI across panels is claimed.',
        temperature_bounds=[.05,20],new_backbone_inference=False,post_hoc=True),indent=2))
    print('Pooled fit:',pd.DataFrame(fitrows).round(6).to_string(index=False),flush=True)
    print('Matched audit-panel native balanced scores:',ag.round(6).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);p.add_argument('--report',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--replicates',type=int,default=1000);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--only',choices=['compact','pooled','all'],default='all');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        full=compact_audit(a.data,a.report,a.out,a.replicates,a.seed) if a.only in ['compact','all'] else None
        if a.only in ['pooled','all']:pooled_panels(a.data,a.report,a.out,full)
