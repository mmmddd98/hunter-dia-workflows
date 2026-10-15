#!/usr/bin/env python3
"""Analyse Heavy-Dimethyl manifests with seed-matched controls (validation only)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from heavy_dimethyl_sweep_common import GUARDRAILS, heavy_macro
from plot_heavy_dimethyl_rt_sweep import generate_heavy_dimethyl_figures

STRATA = {"heavy_nterm": "raw_heavy_heavy_nterm", "heavy_lys": "raw_heavy_heavy_lysine", "heavy_both": "raw_heavy_heavy_both", "nonheavy": "raw_heavy_normal"}
ORGANISM_STRATA = {
 "overall": "raw_minutes", "heavy_nterm": "raw_heavy_heavy_nterm",
 "heavy_lys": "raw_heavy_heavy_lysine", "heavy_both": "raw_heavy_heavy_both",
 "nonheavy": "raw_heavy_normal", "semitryptic": "raw_digestion_semi_tryptic",
}

def args():
 p=argparse.ArgumentParser(); p.add_argument('--sweep-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);return p.parse_args()
def mae(record):
 return record.get('mae') if record and record.get('rows',0)>0 else None
def load(root):
 manifest=json.loads((root/'sweep_manifest.json').read_text()); rows=[]; cells=[]; organism_cells=[]
 for job in manifest['jobs']:
  meta=job['metadata']; metric=Path(job['completion_files'][0])
  if not metric.is_file(): continue
  d=json.loads(metric.read_text()); sources=d.get('sources',{}); vals={}
  for name,key in STRATA.items():
   source_values=[mae(value.get(key,{})) for value in sources.values()]
   vals[f'{name}_sb_mae']=sum(x for x in source_values if x is not None)/len([x for x in source_values if x is not None]) if any(x is not None for x in source_values) else None
   for source,value in sources.items():
    r=value.get(key,{})
    cells.append({**meta,'source':source,'stratum':name,'mae':mae(r),'n':r.get('rows',0) if r.get('rows',0)>0 else None})
  for organism, values in d.get('organisms', {}).items():
   for name, key in ORGANISM_STRATA.items():
    record=values.get(key,{})
    organism_cells.append({**meta,'organism':organism,'stratum':name,'mae':mae(record),'n':record.get('rows',0) if record.get('rows',0)>0 else None})
  vals['overall_mae']=d.get('source_balanced_raw_mae'); vals['semitryptic_mae']=None
  semi=[mae(v.get('raw_digestion_semi_tryptic',{})) for v in sources.values()]; vals['semitryptic_mae']=sum(x for x in semi if x is not None)/len([x for x in semi if x is not None]) if any(x is not None for x in semi) else None
  source_overall=[mae(v.get('raw_minutes',{})) for v in sources.values()]; vals['worst_source_mae']=max(x for x in source_overall if x is not None) if any(x is not None for x in source_overall) else None
  vals['heavy_macro_source_balanced_mae']=heavy_macro(vals); rows.append({**meta,'job_id':job['id'],**vals})
 return pd.DataFrame(rows),pd.DataFrame(cells),pd.DataFrame(organism_cells)
def main():
 a=args(); out=a.output_dir;out.mkdir(parents=True,exist_ok=True); runs,cells,organism_cells=load(a.sweep_dir)
 runs.to_csv(out/'all_runs.csv',index=False); cells.to_csv(out/'source_stratum_metrics.csv',index=False);cells[['model','arm','target','profile','loss','lr','seed','source','stratum','n']].to_csv(out/'source_stratum_counts.csv',index=False)
 organism_cells.to_csv(out/'organism_stratum_metrics.csv',index=False)
 keys=['model','arm','target','policy','profile','loss','lr','weights','oversampling','seed']; controls={}; shared_controls={}
 for _,r in runs.iterrows():
  if (r.arm=='weighting' and r.profile=='W0') or (r.arm=='oversampling' and r.profile=='O0'): controls[(r.arm,r.target,r.loss,r.lr,r.seed)]=r
  if r.profile=='C0': shared_controls[(r.target,r.loss,r.lr,r.seed)]=r
 delta=[]
 for _,r in runs.iterrows():
  c=shared_controls.get((r.target,r.loss,r.lr,r.seed)) if shared_controls else controls.get((r.arm,r.target,r.loss,r.lr,r.seed))
  if c is None or r.profile in ('W0','O0','C0'): continue
  x={k:r[k] for k in keys};
  for m in ['heavy_macro_source_balanced_mae','heavy_nterm_sb_mae','heavy_lys_sb_mae','heavy_both_sb_mae','nonheavy_sb_mae','overall_mae','semitryptic_mae','worst_source_mae']: x['delta_'+m]=r[m]-c[m] if pd.notna(r[m]) and pd.notna(c[m]) else None
  delta.append(x)
 deltas=pd.DataFrame(delta);deltas.to_csv(out/'matched_control_deltas.csv',index=False)
 control_selector = ((organism_cells.arm == 'weighting') & (organism_cells.profile == 'W0')) | ((organism_cells.arm == 'oversampling') & (organism_cells.profile == 'O0')) | (organism_cells.profile == 'C0')
 if (organism_cells.profile == 'C0').any():
  organism_match_keys = ['model','target','policy','loss','lr','seed','organism','stratum']
  controls_by_organism = organism_cells.loc[organism_cells.profile == 'C0', organism_match_keys + ['mae']].rename(columns={'mae':'control_mae'})
 else:
  organism_match_keys = ['model','arm','target','policy','loss','lr','seed','organism','stratum']
  controls_by_organism = organism_cells.loc[control_selector, organism_match_keys + ['mae']].rename(columns={'mae':'control_mae'})
 organism_deltas = organism_cells.loc[~control_selector].merge(controls_by_organism, on=organism_match_keys, how='left')
 organism_deltas['delta_mae'] = organism_deltas['mae'] - organism_deltas['control_mae']
 organism_deltas.to_csv(out/'matched_organism_stratum_deltas.csv',index=False)
 metrics=['heavy_macro_source_balanced_mae','heavy_nterm_sb_mae','heavy_lys_sb_mae','heavy_both_sb_mae','nonheavy_sb_mae','overall_mae','semitryptic_mae','worst_source_mae']
 group=['model','arm','target','policy','profile','loss','lr','weights','oversampling']; summary=runs.groupby(group,dropna=False)[metrics].agg(['mean','std','median','count']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();summary.to_csv(out/'configuration_mean_summary.csv',index=False);runs.to_csv(out/'configuration_seed_summary.csv',index=False)
 if not deltas.empty:
  dmean=deltas.groupby(group,dropna=False).mean(numeric_only=True).reset_index(); merged=summary.merge(dmean,on=group,how='left'); flags=[]
  for _,r in merged.iterrows():
   problems=[]
   if pd.isna(r.get('heavy_macro_source_balanced_mae_mean')): problems.append('FAIL_MISSING_DATA')
   if r.get('delta_heavy_both_sb_mae',0)>0: problems.append('FAIL_HEAVY_BOTH')
   if max(r.get('delta_heavy_nterm_sb_mae',0),r.get('delta_heavy_lys_sb_mae',0))>GUARDRAILS['max_heavy_other_delta']: problems.append('FAIL_HEAVY_OTHER')
   for field,limit,label in [('delta_nonheavy_sb_mae',GUARDRAILS['max_nonheavy_delta'],'NONHEAVY'),('delta_overall_mae',GUARDRAILS['max_overall_delta'],'OVERALL'),('delta_semitryptic_mae',GUARDRAILS['max_semitryptic_delta'],'SEMITRYPTIC'),('delta_worst_source_mae',GUARDRAILS['max_worst_source_delta'],'WORST_SOURCE')]:
    if r.get(field,0)>limit: problems.append('FAIL_'+label)
   flags.append(';'.join(problems) if problems else 'PASS')
  merged['guardrail_status']=flags; merged.sort_values('heavy_macro_source_balanced_mae_mean').to_csv(out/'heavy_macro_ranking.csv',index=False);merged.query("guardrail_status == 'PASS'").to_csv(out/'guardrail_pass_candidates.csv',index=False)
  merged.to_csv(out/'weighting_vs_oversampling.csv',index=False);merged.query("arm == 'weighting'").to_csv(out/'weight_dose_response.csv',index=False);merged.query("arm == 'weighting'").to_csv(out/'weight_lr_interactions.csv',index=False)
 else: merged=summary
 pd.DataFrame().to_csv(out/'embedding_arm_summary.csv',index=False)
 if not deltas.empty:
  model_values = runs['model'].dropna().unique()
  model_label = str(model_values[0]) if len(model_values) == 1 else 'RT model'
  model_label = {'diann': 'DIA-NN', 'carafe2': 'Carafe2'}.get(model_label.lower(), model_label)
  generate_heavy_dimethyl_figures(merged, deltas, out, model_label, organism_deltas)
 print(f'Wrote Heavy-Dimethyl analysis to {out}')
if __name__=='__main__': main()
