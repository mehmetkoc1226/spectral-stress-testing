"""Reanalyse existing images and diagnostic summaries; no model inference.

Run with --data PATH containing stage1_summary.csv and imagenette2-320.tgz,
and --report PATH containing dataset_manifest_imagenette.csv. Optional
stage1_predictions/ holds selected existing restricted prediction CSVs.
"""
from pathlib import Path
import argparse, hashlib, io, itertools, json, tarfile
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kendalltau


def energy(data, report, out):
    archive = data / 'imagenette2-320.tgz'
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    assert digest == '3df6f0d01a2c9592104656642f5e78a3', digest
    manifest = pd.read_csv(report/'dataset_manifest_imagenette.csv')
    lookup = {str(r.relative_path): r for r in manifest.itertuples()}
    axis = np.fft.fftfreq(224)*224
    radius2 = axis[:, None]**2 + axis[None, :]**2
    masks = {}
    for r in [10, 20, 40]:
        masks[f'hard_lp_r{r:03d}_linear'] = (radius2 <= r*r).astype(float)
        masks[f'hard_hp_r{r:03d}_linear'] = 1-masks[f'hard_lp_r{r:03d}_linear']
        if r <= 20:
            masks[f'gauss_lp_r{r:03d}_linear'] = np.exp(-np.log(2)*radius2/(r*r))
            masks[f'gauss_hp_r{r:03d}_linear'] = 1-masks[f'gauss_lp_r{r:03d}_linear']
    matrix = np.stack([m.ravel()**2 for m in masks.values()])
    rows=[]
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or '/val/' not in member.name: continue
            relative = member.name
            if relative not in lookup: continue
            row = lookup[relative]
            im = Image.open(io.BytesIO(tar.extractfile(member).read())).convert('RGB')
            w,h = im.size
            size = (256, int(256*h/w)) if w<=h else (int(256*w/h),256)
            im = im.resize(size, Image.Resampling.BILINEAR)
            left,top = int(round((im.width-224)/2)),int(round((im.height-224)/2))
            arr = np.asarray(im.crop((left,top,left+224,top+224)),dtype=np.float32)/np.float32(255)
            power = np.abs(np.fft.fft2(arr, axes=(0,1)))**2
            power = power.sum(axis=2)
            total = power.sum(); dc = power[0,0]; ac = total-dc
            assert ac > 0
            numerators = matrix @ power.ravel()
            for j,(condition, mask) in enumerate(masks.items()):
                rows.append(dict(global_dataset_index=row.global_dataset_index,
                    target_imagenet1000=row.target_imagenet1000,condition=condition,
                    raw_energy_ratio=numerators[j]/total,
                    ac_energy_ratio=(numerators[j]-mask[0,0]**2*dc)/ac,
                    dc_fraction=dc/total))
    df=pd.DataFrame(rows)
    assert df.global_dataset_index.nunique()==3925, df.shape
    df.to_csv(out/'energy_per_image.csv.gz',index=False,compression='gzip')
    aggregate=df.groupby('condition')[['raw_energy_ratio','ac_energy_ratio','dc_fraction']].agg(['mean','std'])
    aggregate.columns=['_'.join(c) for c in aggregate.columns]
    summary=pd.read_csv(data/'stage1_summary.csv')
    old=summary[(summary.dataset=='imagenette')&(summary.protocol=='imagenet1000')].groupby('condition').mean_energy_ratio.mean()
    aggregate['archived_raw_energy_ratio']=old
    aggregate['raw_reproduction_error']=aggregate.raw_energy_ratio_mean-aggregate.archived_raw_energy_ratio
    aggregate.to_csv(out/'energy_summary.csv')
    df.groupby(['target_imagenet1000','condition'])[['raw_energy_ratio','ac_energy_ratio']].mean().to_csv(out/'energy_by_class.csv')
    (out/'energy_provenance.json').write_text(json.dumps(dict(archive_md5=digest,n_images=3925,
        geometry='PIL RGB; shorter edge 256 bilinear; round-centred 224 crop; float32 /255',
        calculation='Parseval, RGB-channel summed DFT power; DC omitted in numerator and denominator for AC',
        new_inference=False,max_raw_reproduction_error=float(aggregate.raw_reproduction_error.abs().max())),indent=2))
    print('Energy audit:',aggregate.round(7).to_string(),flush=True)


def ranks(data,out):
    df=pd.read_csv(data/'stage1_summary.csv')
    assert len(df)==432 and not df.duplicated(['dataset','model','condition','protocol']).any()
    df[['dataset','model','condition','protocol']].to_csv(out/'diagnostic_condition_ledger.csv',index=False)
    rows=[]
    for dataset,ds in df.groupby('dataset'):
        profiles={}
        for (condition,protocol),frame in ds.groupby(['condition','protocol']):
            if len(frame)==6: profiles[(condition,protocol)]=frame.set_index('model').accuracy.sort_index()
        for (ka,a),(kb,b) in itertools.combinations(profiles.items(),2):
            assert list(a.index)==list(b.index)
            av,bv=a.to_numpy(),b.to_numpy()
            discordant=concordant=ties=0
            for i,j in itertools.combinations(range(6),2):
                prod=np.sign(av[i]-av[j])*np.sign(bv[i]-bv[j])
                discordant+=int(prod<0);concordant+=int(prod>0);ties+=int(prod==0)
            rows.append(dict(dataset=dataset,condition_a=ka[0],protocol_a=ka[1],
                condition_b=kb[0],protocol_b=kb[1],n_models=6,tau_b=kendalltau(av,bv).statistic,
                strict_inversions=discordant,concordant_pairs=concordant,pairs_tied_in_either=ties,
                total_model_pairs=15))
    pd.DataFrame(rows).to_csv(out/'checkpoint_rank_comparisons.csv',index=False)
    print('Condition ledger and',len(rows),'descriptive ranking comparisons saved',flush=True)


def collapse(data,out):
    rows=[]
    for path in (data/'stage1_predictions').rglob('*.csv.gz'):
        df=pd.read_csv(path)
        prediction=next((c for c in ['prediction','pred','predicted_class','prediction_imagenet1000'] if c in df),None)
        if prediction is None: raise ValueError(df.columns.tolist())
        counts=df[prediction].value_counts().to_numpy(); q=counts/counts.sum()
        rows.append(dict(dataset=path.parent.parent.name,model=path.parent.name,
            condition=path.name.split('__')[0],n=len(df),effective_class_count=float(np.exp(-(q*np.log(q)).sum())),
            inverse_simpson=float(1/np.sum(q*q)),occupied_classes=len(counts)))
    if rows:
        frame=pd.DataFrame(rows);frame.to_csv(out/'effective_classes_by_checkpoint.csv',index=False)
        ag=frame.groupby(['dataset','condition'])[['effective_class_count','inverse_simpson','occupied_classes']].mean()
        ag.to_csv(out/'effective_classes_summary.csv');print(ag.round(4).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--only',choices=['energy','ranks','collapse','all'],default='all');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.only in ['energy','all']:energy(a.data,a.report,a.out)
    if a.only in ['ranks','all']:ranks(a.data,a.out)
    if a.only in ['collapse','all']:collapse(a.data,a.out)
