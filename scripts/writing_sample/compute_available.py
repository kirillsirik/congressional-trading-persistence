"""A2/A5 and identifiable supporting checks, with one frozen numerical results file.

No restricted inputs are reconstructed or replaced with assumed statistics.
Run: python scripts/writing_sample/compute_available.py
Writes its rows to data/v4/writing_sample/components/available.csv and rebuilds results.csv from all components.
"""
from pathlib import Path
import sys,hashlib,json,csv,zipfile,collections,platform
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/coherence'))
from aggregate_checks import moments,jackknife,explicit_moments,validate_means,unit_roots
sys.path.insert(0,str(Path(__file__).resolve().parent))
import merge_results
RUN=ROOT/'data/v4/round2/closeout/closeout_2026-11-10'
OUT=ROOT/'data/v4/writing_sample'


def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[];checks=[]
    def add(analysis,clock,scale,term,statistic,value,unit='',inference='',source='',lambda_value=np.nan):
        records.append(dict(analysis=analysis,clock=clock,scale=scale,term=term,statistic=statistic,lambda_value=lambda_value,value=float(value),unit=unit,inference=inference,source=source))
    def check(name,a,b):
        delta=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        ok=bool(np.allclose(a,b,atol=1e-7,rtol=2e-10));checks.append(dict(check=name,passed=ok,max_abs_gap=delta))
        if not ok:raise AssertionError((name,delta))
    means=pd.read_csv(RUN/'inputs/principal_household_means.csv');point=pd.read_csv(RUN/'paired/point_statistics.csv');boot=pd.read_csv(RUN/'paired/paired_resamples.csv');loo_stored=pd.read_csv(RUN/'paired/leave_one_household_out.csv');eff=pd.read_csv(RUN/'small_cluster/small_cluster_inference.csv')
    terms=['K11','K00','KDD','K0D','KD0','difference'];source='data/v4/round2/closeout/closeout_2026-11-10/'
    for (clock,scale),g in means.groupby(['clock','scale']):
        g=validate_means(g);p=moments(g)[0];stored=point[(point.clock==clock)&(point.scale==scale)].iloc[0]
        check(f'{clock}/{scale} points',p,stored[terms].to_numpy(float));check(f'{clock}/{scale} independent least squares',p,explicit_moments(g))
        multiplicities=pd.read_csv(RUN/f'paired/selection_multiplicities_{clock}.csv')
        if multiplicities.columns[1:].tolist()!=[f'hh_{i}' for i in g.household_index]:raise ValueError('Household bootstrap order differs')
        b=moments(g,multiplicities.iloc[:,1:].to_numpy());stored_b=boot[(boot.clock==clock)&(boot.scale==scale)].sort_values('resample')[terms].to_numpy(float)
        check(f'{clock}/{scale} all original bootstrap vectors',b,stored_b)
        coefs=[p[1],p[3]+p[4],p[2]]
        for term,v in zip(['constant','linear','quadratic'],coefs):add('A2',clock,scale,term,'coefficient',v,'bp2','exact affine interpolation',source+'paired/point_statistics.csv')
        roots=unit_roots(*coefs)
        add('A2',clock,scale,'curve','roots_in_unit_interval',len(roots),'count',source=source+'paired/point_statistics.csv')
        for j,v in enumerate(roots):add('A2',clock,scale,'curve',f'root_{j+1}',v,'fraction','exact affine interpolation',source+'paired/point_statistics.csv')
        for lam in np.linspace(0,1,101):
            est=coefs[0]+lam*coefs[1]+lam*lam*coefs[2];bs=b[:,1]+lam*(b[:,3]+b[:,4])+lam*lam*b[:,2]
            for stat,v in [('estimate',est),('ci_low',np.quantile(bs,.025)),('ci_high',np.quantile(bs,.975))]:add('A2',clock,scale,'curve',stat,v,'bp2','4999 paired household resamples; pointwise percentile band',source+'paired/paired_resamples.csv',lam)
            if np.isclose(lam,[0,.25,.5,.75,1]).any():
                z=g.copy();z.Y1_A=g.Y0_A+lam*(g.Y1_A-g.Y0_A);z.Y1_B=g.Y0_B+lam*(g.Y1_B-g.Y0_B)
                check(f'{clock}/{scale} direct interpolation at {lam}',est,moments(z)[0,0])
        p,deleted,cov,se,lo,hi=jackknife(g)
        old=loo_stored[(loo_stored.clock==clock)&(loo_stored.scale==scale)].sort_values('omitted_household_index')[terms].to_numpy(float)
        check(f'{clock}/{scale} all original leave-one-out vectors',deleted,old)
        G=len(g);Gstar=float(eff[(eff.clock==clock)&(eff.specification=='principal')&(eff.working_model=='inverse_weights')].G_star_rho1.iloc[0]);df_eff=Gstar-1
        for term in ['difference','KDD']:
            j=terms.index(term)
            vals={'estimate':p[j],'se_jackknife':se[j],'households':G,'df_households':G-1,'ci_low_households':lo[j],'ci_high_households':hi[j],'effective_clusters':Gstar,'df_effective_heuristic':df_eff,'critical_effective_heuristic':t.ppf(.975,df_eff),'ci_low_effective_heuristic':p[j]-t.ppf(.975,df_eff)*se[j],'ci_high_effective_heuristic':p[j]+t.ppf(.975,df_eff)*se[j]}
            for stat,v in vals.items():
                unit='count' if stat in ['households','effective_clusters'] else ('unitless' if 'df_' in stat or 'critical_' in stat else 'bp2')
                inference='Heuristic critical value only; effective clusters from another statistic do not calibrate this jackknife' if 'effective' in stat else 'Paired household delete-one jackknife; refit both activity projections; approximate t(G-1) interval'
                add('A5',clock,scale,term,stat,v,unit,inference,source+'paired/leave_one_household_out.csv; '+source+'small_cluster/small_cluster_inference.csv')
        if scale=='raw':
            for bench in ['Y1','Y0']:
                mean=float((g.n_A*g[f'{bench}_A']+g.n_B*g[f'{bench}_B']).sum()/(g.n_A.sum()+g.n_B.sum()))
                add('A4_existing_mean',clock,scale,bench,'mean',mean,'bp','mean only; same two-way SE not recoverable',source+'inputs/principal_household_means.csv')
        if scale=='winsorized':
            w=np.sqrt(g.n_A.to_numpy(float)*g.n_B.to_numpy(float));ihh=w.sum()**2/(w@w)
            add('weight_check',clock,scale,'geometric_weights','inverse_herfindahl',ihh,'households',source=source+'inputs/principal_household_means.csv')
    hh=pd.read_csv(RUN/'inputs/household_test_statistics.csv')
    for clock,g in hh.groupby('clock'):
        for term,v in [('households',len(g)),('negative_means',(g.mean_bp<0).sum()),('two_sided_rejections',(g.p_two_sided<.05).sum()),('negative_rejections',((g.p_two_sided<.05)&(g.mean_bp<0)).sum())]:add('existing_household_tests',clock,'winsorized',term,'count',v,'households',source=source+'inputs/household_test_statistics.csv')
    old=pd.read_csv(ROOT/'data/v4/results.csv')
    rows=old[(old.analysis=='robustness')&old.variant.isin(['sales','signed_all'])&(old.statistic=='tau2')]
    for _,r in rows.iterrows():
        for stat,v in [('estimate',r.value),('ci_low',r.boot_lo),('ci_high',r.boot_hi),('se_clustered',r.se_block),('purchases_or_trades',r.n_events),('households',r.n_households)]:add('A3_existing_event_study',r.clock,'winsorized',r.variant,stat,v,'count' if stat in ['purchases_or_trades','households'] else 'bp2','existing event-study result; factor-only counterpart absent','data/v4/results.csv')
    component=OUT/'components'/'available.csv';component.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(component,index=False,float_format='%.17g',lineterminator='\n')
    merge_results.merge(OUT/'results.csv',component.parent,require_all=False)
    inputs=[RUN/'inputs/principal_household_means.csv',RUN/'paired/point_statistics.csv',RUN/'paired/paired_resamples.csv',RUN/'paired/leave_one_household_out.csv',RUN/'small_cluster/small_cluster_inference.csv',RUN/'inputs/household_test_statistics.csv',ROOT/'data/v4/results.csv']
    inputs +=[RUN/f'paired/selection_multiplicities_{c}.csv' for c in ['T','P']]
    inputs +=[ROOT/'scripts/coherence/aggregate_checks.py']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(script_sha256=sha(Path(__file__)),inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},component_sha256=sha(component),results_sha256=sha(OUT/'results.csv'),environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),checks=checks,all_passed=all(x['passed'] for x in checks),new_simulations=False,A2='Existing paired household resamples only; no new random draws; percentile bands pointwise, not simultaneous; transformed endpoint interpolation not daily alpha scaling',A5='Gstar is the principal event-study slope effective-cluster diagnostic under inverse_weights and rho=1. Reusing Gstar-1 for the gap and KDD is only a requested critical-value sensitivity, not calibrated degrees of freedom.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    frame=pd.DataFrame(records);print(frame[(frame.analysis=='A5')&(frame.scale=='winsorized')].pivot(index=['clock','term'],columns='statistic',values='value').to_string());print('Passed',len(checks),'grouped checks')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__);ap.parse_args()
    main()
