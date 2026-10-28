"""Validate completed Colab summaries and build manuscript tables and Figure 16.

Reads immutable exported run/analysis outputs. No fitting, inference, or
selection on evaluation images is performed here.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'figures').mkdir(exist_ok=True)
DATA=ROOT/'analysis_results/strengthening'
TABLES=DATA/'tables';TABLES.mkdir(exist_ok=True)
C=pd.read_csv(DATA/'controls/CONTROL_PAIRED_ACCURACY.csv')
S=pd.read_csv(DATA/'controls/summary.csv')
P=pd.read_csv(DATA/'calibration/CALIBRATION_PAIRED_CONTRASTS.csv')
B=pd.read_csv(DATA/'calibration/CALIBRATION_BALANCED_SCORES.csv')
Q=pd.read_csv(DATA/'calibration/summary.csv')
REPLAY=pd.read_csv(DATA/'calibration/REPLAY_AGREEMENT.csv')
MODELS=['resnet50_a1','swin_t_in1k','vitb16_in1k']
NAMES={'resnet50_a1':'ResNet-50 A1','swin_t_in1k':'Swin-T','vitb16_in1k':'ViT-B/16'}
FAMILIES=['unseen_radius','unseen_filter','natural_corruption','unseen_dataset']
def family(ds,c):
    if ds=='imagenet_a':return 'unseen_dataset'
    if c=='clean_common':return 'heldout_clean'
    if c.startswith('gauss'):return 'unseen_filter'
    if c.startswith(('blur','noise')):return 'natural_corruption'
    return 'seen_spectral' if '_r020_' in c else 'unseen_radius'

assert len(C)==96 and len(P)==108 and len(B)==30
assert len(S)==126 and len(Q)==780
assert not S.duplicated(['dataset','model','condition','protocol','method']).any()
assert not Q.duplicated(['dataset','model','condition','protocol','method']).any()
for task,njobs,nforwards in [('controls',63,247275),('calibration',78,335790)]:
    t=pd.read_csv(DATA/task/'timing.csv');complete=json.loads((DATA/task/'RUN_COMPLETE.json').read_text())
    assert len(t)==njobs and int(t.n.sum())==nforwards==complete['image_forwards']
    protocol=json.loads((DATA/task/'PROTOCOL.json').read_text())
    assert protocol['mode']=='full' and protocol['task']==task and not protocol['curve_grid']
    for key,file in [('runner_sha256','run_strengthening.py'),('controls_sha256','spectral_controls.py')]:
        assert hashlib.sha256((ROOT/'strengthening_experiments'/file).read_bytes()).hexdigest()==protocol[key]
for _,r in C.iterrows():
    def acc(condition):return S[(S.model==r.model)&(S.condition==condition)&(S.protocol==r.protocol)].accuracy.item()
    assert np.isclose(r.estimate,acc(r.condition_a)-acc(r.condition_b),atol=1e-12)
Q['family']=[family(ds,c) for ds,c in zip(Q.dataset,Q.condition)]
native=Q[Q.protocol=='imagenet1000']
for _,r in B.iterrows():
    subset=native[(native.method==r.method)&(native.family.isin(FAMILIES) if r.scope=='all' else native.family.eq(r.scope))]
    for metric in ['nll','ece15','brier']:
        actual=subset.groupby(['model','family'])[metric].mean().groupby('model').mean().mean()
        assert np.isclose(actual,r[metric],atol=1e-12)
for _,r in P.iterrows():
    a=B[(B.scope==r.scope)&(B.method==r.method_a)][r.metric].item()
    b=B[(B.scope==r.scope)&(B.method==r.method_b)][r.metric].item()
    assert np.isclose(r.estimate,a-b,atol=1e-12)
assert native.groupby(['dataset','model','condition']).accuracy.nunique().max()==1
assert Q.groupby(['dataset','model','condition','protocol']).accuracy.nunique().max()==1

def effect(r,scale=1,digits=3):
    a,b,c=[scale*r[k] for k in ['estimate','ci_low','ci_high']]
    return f'${a:+.{digits}f}$ [${b:+.{digits}f},{c:+.{digits}f}$]'
rows=[]
for model in MODELS:
    for mask in ['hard','gauss']:
        for radius in [10,20]:
            g=C[(C.model==model)&(C.protocol=='imagenet1000')&C.condition_a.str.startswith(f'{mask}_hp_r{radius:03d}')]
            a=g[g.contrast=='signed_hp_minus_lp'].iloc[0]
            b=g[g.contrast=='rms_hp_minus_lp'].iloc[0]
            d=g[g.contrast=='dc_restore_hp'].iloc[0]
            rows.append(' & '.join([NAMES[model],('Hard' if mask=='hard' else 'Gaussian'),str(radius),effect(a,100,1),effect(b,100,1),effect(d,100,1)])+r'\\')
(TABLES/'mean_contrast_rows.tex').write_text('\n'.join(rows)+'\n')
methods=['clean_ts','pooled_ts','fcts','confidence_ts']
labels=['Radius','Gaussian filter','Blur/noise','ImageNet-A','Balanced unseen','Held-out clean']
rows=[]
for scope,label in zip(FAMILIES+['all','heldout_clean'],labels):
    rows.append(label+' & '+' & '.join(f'{B[(B.scope==scope)&(B.method==m)].nll.item():.3f}' for m in methods)+r'\\')
(TABLES/'full_cohort_family_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for m,label in [('fcts','FCTS'),('confidence_ts','CCTS'),('pooled_ts','Pooled scalar TS')]:
    a=P[(P.scope=='heldout_clean')&(P.method_a==m)&(P.method_b=='clean_ts')&(P.metric=='nll')].iloc[0]
    b=P[(P.scope=='all')&(P.method_a==m)&(P.method_b=='clean_ts')&(P.metric=='nll')].iloc[0]
    rows.append(label+' & '+effect(a,digits=4)+' & '+effect(b)+r'\\')
(TABLES/'clean_shift_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for m,label in [('clean_ts','Clean scalar TS'),('pooled_ts','Pooled scalar TS'),('fcts','FCTS'),('confidence_ts','CCTS')]:
    b=B[(B.scope=='all')&(B.method==m)].iloc[0]
    clean=B[(B.scope=='heldout_clean')&(B.method==m)].nll.item()
    rows.append(label+' & '+' & '.join(f'{v:.3f}' for v in [b.nll,b.ece15,b.brier,clean])+r'\\')
(TABLES/'pooled_full_scores_rows.tex').write_text('\n'.join(rows)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10.4,5.6),sharey=True)
settings=[(model,mask,r) for r in [10,20] for model in MODELS for mask in ['hard','gauss']]
ticklabels=[NAMES[m]+' / '+('hard' if k=='hard' else 'Gaussian') for m,k,r in settings]
for ax,protocol,title in zip(axes,['imagenet1000','restricted10'],['Native-1000','Restricted-10']):
    for i,(model,mask,radius) in enumerate(settings):
        g=C[(C.model==model)&(C.protocol==protocol)&C.condition_a.str.startswith(f'{mask}_hp_r{radius:03d}')]
        for contrast,offset,color,marker in [('signed_hp_minus_lp',-.13,'#6C7A89','o'),('rms_hp_minus_lp',.13,'#007E87','s')]:
            row=g[g.contrast==contrast].iloc[0];a,l,h=100*row[['estimate','ci_low','ci_high']].to_numpy(float)
            ax.errorbar(a,i+offset,xerr=[[a-l],[h-a]],fmt=marker,ms=4,color=color,capsize=2,lw=1.1,
                label=('Signed linear' if contrast=='signed_hp_minus_lp' else 'Mean + AC-RMS matched') if i==0 else None)
    ax.axvline(0,color='#37495B',lw=.8);ax.axhline(5.5,color='#7E8995',lw=.8,ls='--')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set(xlim=(-62,55),xticks=[-60,-40,-20,0,20,40],yticks=range(12),yticklabels=ticklabels,ylim=(11.7,-.7))
    ax.set_title(title,fontsize=11,weight='bold',pad=10);ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('HP minus LP accuracy (percentage points)')
axes[0].text(-.66,.77,'r = 10',transform=axes[0].transAxes,rotation=90,va='center',fontweight='bold')
axes[0].text(-.66,.25,'r = 20',transform=axes[0].transAxes,rotation=90,va='center',fontweight='bold')
handles,leglabels=axes[0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.58,0))
fig.tight_layout(rect=(0,.07,1,1),w_pad=2)
for ext in ['pdf','png','svg']:fig.savefig(ROOT/'figures'/f'Figure_16.{ext}',dpi=220)
plt.close(fig)

audit=dict(status='passed',controls_jobs=63,controls_image_forwards=247275,controls_summary_rows=126,
    controls_contrasts=96,calibration_jobs=78,calibration_image_forwards=335790,calibration_summary_rows=780,
    calibration_paired_contrasts=108,calibration_balanced_scores=30,
    all_contrast_points_reconciled_with_summaries=True,all_balanced_points_recomputed=True,
    new_replay_accuracy_invariant_across_calibrators=True,
    saved_panel_replay_comparisons=int(REPLAY.n_audit.sum()),
    saved_panel_replay_prediction_mismatches=int(REPLAY.prediction_mismatches.sum()),
    saved_panel_max_abs_logit_difference=float(REPLAY.max_abs_logit_difference.max()),
    note='Cross-run replay differences are distinct from within-run calibration prediction invariance; original diagnostic results remain archived unchanged.')
(DATA/'LOCAL_VALIDATION.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
