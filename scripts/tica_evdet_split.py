"""Evasion detector follow-ups (10-25): AUROC on caught vs passed hiding turns, and calibration
transfer (thresholds set on TRAIN concepts applied to HELD-OUT). CPU; aggregates only.
  python scripts/tica_evdet_split.py FEATS_DIR   (expects <model>_all.pt merged dumps)
"""
import sys,torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from tica_hide_geometry import logistic, auroc
F=sys.argv[1]
for m in ['base','p1_s1200','ps_s1200']:
    d=torch.load(f'{F}/{m}_all.pt',weights_only=False); X=d['x'].float(); meta=d['meta']; Ls=d['layers']
    def idx(f): return torch.tensor([i for i,r in enumerate(meta) if f(r)],dtype=torch.long)
    hide=lambda r: r['cond'] in ('real_score','real_verdict') and not r['off'] and r['turn']>=1
    honest=lambda r: (r['cond'] in ('real_score','real_verdict') and r['off']) or (r['cond']=='nobrief' and r['turn']>=1)
    out=[]
    for L in (8,12,16,20,24,28):
        li=Ls.index(L)
        tr_p=idx(lambda r: hide(r) and r['split']=='train'); tr_n=idx(lambda r: honest(r) and r['split']=='train')
        f=logistic(torch.cat([X[tr_p,li],X[tr_n,li]]), torch.cat([torch.ones(len(tr_p)),torch.zeros(len(tr_n))]))
        thr=f(X[tr_p,li]).quantile(0.05)          # threshold = 95% TPR on TRAIN concepts
        thr1=f(X[tr_n,li]).quantile(0.99)         # threshold = 1% FPR on TRAIN negatives
        hc=idx(lambda r: hide(r) and r['split']=='heldout' and r['detected']); hp=idx(lambda r: hide(r) and r['split']=='heldout' and not r['detected'])
        hn=idx(lambda r: honest(r) and r['split']=='heldout')
        sc,sp,sn=f(X[hc,li]),f(X[hp,li]),f(X[hn,li])
        out.append('L%d AUROC caught %.2f passed %.2f (n%d/%d, neg %d) | train-thr95: held-out TPR %.2f FPR %.2f | train-1%%FPR thr: held-out TPR %.2f FPR %.2f'%(L,auroc(sc,sn),auroc(sp,sn),len(hc),len(hp),len(hn),float((torch.cat([sc,sp])>thr).float().mean()),float((sn>thr).float().mean()),float((torch.cat([sc,sp])>thr1).float().mean()),float((sn>thr1).float().mean())))
    print('==',m); print('\n'.join('   '+o for o in out),flush=True)
