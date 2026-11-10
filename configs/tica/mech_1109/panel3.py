import json,os
R='/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/'
M={'9B':'final_v4d_k03_tamia_9b','4B':'final_v4d_k03_tamia_4b','2B':'final_v4d_tamia_2b_x1800'}
for sp in ('train','heldout'):
  for s in (0,1,2,3):
    row=[]
    for m,r in M.items():
      p=f'{R}{r}/quality/step_1800/panel_{sp}.s{s}.json'
      if not os.path.exists(p): row.append(f'{m} --'); continue
      d=json.load(open(p)); P=d['aggregate']['panel']
      dets={k:v['det'] for k,v in P.items() if isinstance(v,dict) and v.get('det') is not None}
      mean=sum(dets.values())/len(dets); beat=sum(v<0.5 for v in dets.values())
      row.append(f"{m} det {mean:.2f} beaten {beat}/{len(dets)} fails {len(d.get('panel_failed',{}))}")
    print(f'{sp} s{s}: '+' | '.join(row))
