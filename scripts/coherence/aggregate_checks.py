"""Reproduce frozen household moments; compute paired sensitivity checks from exports.

Run from the repository root: python scripts/coherence/aggregate_checks.py
No restricted inputs, original outputs, or manuscript files are modified.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'data/v4/round2/closeout/closeout_2026-11-10'
OUT = ROOT / 'data/v4/coherence'
TERMS = ['K11', 'K00', 'KDD', 'K0D', 'KD0', 'difference']


def validate_means(g):
    required = ['household_index','n_A','n_B','Y1_A','Y1_B','Y0_A','Y0_B']
    if not set(required).issubset(g):
        raise ValueError('Missing required household-mean columns')
    if g.household_index.duplicated().any():
        raise ValueError('Duplicate household keys within clock and scale')
    if not np.isfinite(g[required].to_numpy(float)).all():
        raise ValueError('Household means and counts must be finite')
    n=g[['n_A','n_B']].to_numpy(float)
    if (n<1).any() or (n!=np.floor(n)).any():
        raise ValueError('Purchase counts must be positive integers')
    return g.sort_values('household_index').reset_index(drop=True)


def partial_cov(a,b,c,W):
    """Weighted partial covariance with a constant and one activity control.

    Each row of W is a full set of nonnegative household weights, possibly
    multiplied by household bootstrap multiplicities. This Schur complement
    is checked independently against explicit least-squares residuals.
    """
    W=np.atleast_2d(np.asarray(W,float))
    if (W<0).any() or (W.sum(axis=1)<=0).any():
        raise ValueError('Weights must be nonnegative and sum to a positive number')
    W=W/W.sum(axis=1,keepdims=True)
    ma,mb,mc=W@a,W@b,W@c
    vc=W@(c*c)-mc*mc
    if (vc<=1e-12).any():
        raise ValueError('Activity projection is rank deficient')
    return W@(a*b)-ma*mb-(W@(a*c)-ma*mc)*(W@(b*c)-mb*mc)/vc


def moments(g,multiplicity=None,design=None):
    g=validate_means(g)
    w,c=(np.sqrt(g.n_A.to_numpy(float)*g.n_B.to_numpy(float)),np.log1p(g.n_A.to_numpy(float))) if design is None else design
    M=np.ones((1,len(g))) if multiplicity is None else np.atleast_2d(multiplicity)
    a,b,x,y=[g[k].to_numpy(float) for k in ['Y1_A','Y1_B','Y0_A','Y0_B']]
    da,db=a-x,b-y
    v=np.column_stack([partial_cov(u,v,c,M*w) for u,v in [(a,b),(x,y),(da,db),(x,db),(da,y)]])
    return np.column_stack([v,v[:,0]-v[:,1]])


def explicit_moments(g):
    g=validate_means(g);w=np.sqrt(g.n_A.to_numpy(float)*g.n_B.to_numpy(float));c=np.log1p(g.n_A.to_numpy(float))
    C=np.column_stack([np.ones(len(g)),c]);X=g[['Y1_A','Y1_B','Y0_A','Y0_B']].to_numpy(float)
    R=X-C@np.linalg.lstsq(C*np.sqrt(w)[:,None],X*np.sqrt(w)[:,None],rcond=None)[0]
    a,b,x,y=R.T;da,db=a-x,b-y;wn=w/w.sum()
    v=np.array([np.sum(wn*u*v) for u,v in [(a,b),(x,y),(da,db),(x,db),(da,y)]])
    return np.r_[v,v[0]-v[1]]


def jackknife(g):
    g=validate_means(g);G=len(g);point=moments(g)[0]
    deleted=np.stack([moments(g.drop(index=j))[0] for j in range(G)])
    centered=deleted-deleted.mean(axis=0)
    covariance=(G-1)/G*(centered.T@centered)
    se=np.sqrt(np.diag(covariance));crit=t.ppf(.975,G-1)
    return point,deleted,covariance,se,point-crit*se,point+crit*se


def unit_roots(c0,c1,c2):
    roots=np.roots([c2,c1,c0]) if abs(c2)>1e-12 else np.roots([c1,c0])
    return sorted(float(r.real) for r in roots if abs(r.imag)<1e-10 and 0<=r.real<=1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    means=pd.read_csv(RUN/'inputs/principal_household_means.csv')
    old=pd.read_csv(RUN/'paired/paired_resamples.csv')
    point_old=pd.read_csv(RUN/'paired/point_statistics.csv')
    manifest=json.loads((RUN/'inputs/exports_manifest.json').read_text())
    if not manifest['household_index'].startswith('global anonymous key'):
        raise ValueError('Cross-clock identifier stability is not documented')
    checks=[];jack=[];deletions=[];jcov=[];poly=[];curves=[];roots=[];root_summary=[]
    def check(name,a,b):
        gap=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        ok=bool(np.allclose(a,b,rtol=2e-10,atol=1e-7))
        checks.append(dict(check=name,passed=ok,max_abs_difference=gap))
        if not ok:raise AssertionError((name,gap))
    for (clock,scale),g in means.groupby(['clock','scale']):
        g=validate_means(g);point=moments(g)[0]
        p=point_old[(point_old.clock==clock)&(point_old.scale==scale)].iloc[0]
        check(f'{clock}/{scale}: frozen point statistics',point,p[TERMS].to_numpy(float))
        check(f'{clock}/{scale}: independent QR residualization',point,explicit_moments(g))
        mult=pd.read_csv(RUN/f'paired/selection_multiplicities_{clock}.csv')
        if list(mult.columns[1:])!=[f'hh_{x}' for x in g.household_index]:
            raise ValueError('Stored bootstrap multiplicities do not align with household order')
        boot=moments(g,mult.iloc[:,1:].to_numpy())
        stored=old[(old.clock==clock)&(old.scale==scale)].sort_values('resample')[TERMS].to_numpy(float)
        check(f'{clock}/{scale}: all 4999 paired bootstrap vectors',boot,stored)
        check(f'{clock}/{scale}: all decomposition identities',boot[:,0],boot[:,1:5].sum(axis=1))
        point,loo,cov,se,lo,hi=jackknife(g)
        for j,term in enumerate(TERMS):
            jack.append(dict(clock=clock,scale=scale,term=term,estimate=point[j],se_jackknife=se[j],df=len(g)-1,ci_low=lo[j],ci_high=hi[j],deleted_min=loo[:,j].min(),deleted_max=loo[:,j].max()))
            for k,term2 in enumerate(TERMS):jcov.append(dict(clock=clock,scale=scale,term1=term,term2=term2,covariance=cov[j,k]))
        for h,vals in zip(g.household_index,loo):deletions.append(dict(clock=clock,scale=scale,omitted_household_index=h,**dict(zip(TERMS,vals))))
        c0,c1,c2=point[1],point[3]+point[4],point[2]
        rt=unit_roots(c0,c1,c2)
        poly.append(dict(clock=clock,scale=scale,constant=c0,linear=c1,quadratic=c2,roots_in_unit_interval=';'.join(f'{x:.15g}' for x in rt)))
        for lam in np.linspace(0,1,101):
            b=boot[:,1]+lam*(boot[:,3]+boot[:,4])+lam*lam*boot[:,2]
            estimate=c0+lam*c1+lam*lam*c2
            # Direct calculation proves identity at interior values, independent of table rounding.
            if lam in [0,.25,.5,.75,1]:
                z=g.copy();z['Y1_A']=g.Y0_A+lam*(g.Y1_A-g.Y0_A);z['Y1_B']=g.Y0_B+lam*(g.Y1_B-g.Y0_B)
                check(f'{clock}/{scale}: interpolation lambda={lam}',estimate,moments(z)[0,0])
            curves.append(dict(clock=clock,scale=scale,lambda_=lam,estimate=estimate,ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975),bootstrap_share_positive=(b>0).mean(),bootstrap_n=len(b)))
        for j,v in enumerate(boot):
            r=unit_roots(v[1],v[3]+v[4],v[2]);roots.append(dict(clock=clock,scale=scale,resample=j,number_roots=len(r),roots_in_unit_interval=';'.join(f'{x:.15g}' for x in r),negative_at_zero=v[1]<0,positive_at_one=v[0]>0))
        rr=[x for x in roots if x['clock']==clock and x['scale']==scale]
        root_summary.append(dict(clock=clock,scale=scale,bootstrap_n=len(rr),no_root=sum(x['number_roots']==0 for x in rr),one_root=sum(x['number_roots']==1 for x in rr),two_roots=sum(x['number_roots']==2 for x in rr),negative_zero_positive_one=sum(x['negative_at_zero'] and x['positive_at_one'] for x in rr)))
    common=[];common_diff=[];common_design=[];common_boot=[]
    seed=20261111;B=19999
    for scale in ['raw','winsorized']:
        T=validate_means(means[(means.clock=='T')&(means.scale==scale)])
        P=validate_means(means[(means.clock=='P')&(means.scale==scale)])
        keys=sorted(set(T.household_index)&set(P.household_index));T=validate_means(T[T.household_index.isin(keys)]);P=validate_means(P[P.household_index.isin(keys)])
        assert T.household_index.tolist()==P.household_index.tolist()
        rng=np.random.default_rng(seed);sel=rng.integers(0,len(keys),size=(B,len(keys)))
        M=np.zeros((B,len(keys)),dtype=int);np.add.at(M,(np.arange(B)[:,None],sel),1)
        designs={k:(np.sqrt(g.n_A.to_numpy(float)*g.n_B.to_numpy(float)),np.log1p(g.n_A.to_numpy(float))) for k,g in [('T',T),('P',P)]}
        for design in ['native_clock_specific','fixed_trade_design','fixed_disclosure_design']:
            stats={};bs={}
            for clock,g in [('T',T),('P',P)]:
                source=clock if design=='native_clock_specific' else ('T' if design=='fixed_trade_design' else 'P')
                d=designs[source];v=moments(g,design=d)[0];bv=moments(g,M,design=d);stats[clock]=v;bs[clock]=bv
                for term,j in zip(TERMS,range(len(TERMS))):
                    q=np.quantile(bv[:,j],[.025,.975])
                    common.append(dict(clock=clock,scale=scale,design=design,design_source=source,households=len(g),purchases_A=int(g.n_A.sum()),purchases_B=int(g.n_B.sum()),term=term,estimate=v[j],ci_low=q[0],ci_high=q[1],bootstrap_n=B))
                if scale=='winsorized':
                    for h,w,c in zip(g.household_index,*d):common_design.append(dict(clock=clock,design=design,household_index=h,weight=w,activity_log=c))
            for term,j in zip(TERMS,range(len(TERMS))):
                bv=bs['P'][:,j]-bs['T'][:,j];q=np.quantile(bv,[.025,.975])
                common_diff.append(dict(scale=scale,design=design,term=term,disclosure_minus_trade=stats['P'][j]-stats['T'][j],ci_low=q[0],ci_high=q[1],bootstrap_n=B))
    outputs={'baseline_checks':checks,'paired_jackknife':jack,'paired_jackknife_covariance':jcov,'paired_leave_one_out':deletions,'affine_interpolation_coefficients':poly,'affine_interpolation_curve':curves,'affine_interpolation_bootstrap_roots':roots,'affine_interpolation_root_summary':root_summary,'common_household_statistics':common,'common_household_clock_differences':common_diff,'common_household_design':common_design}
    for name,rows in outputs.items():pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False,float_format='%.17g')
    files=[RUN/'inputs/principal_household_means.csv',RUN/'inputs/exports_manifest.json',RUN/'paired/paired_resamples.csv',RUN/'paired/point_statistics.csv',RUN/'paired/selection_multiplicities_T.csv',RUN/'paired/selection_multiplicities_P.csv']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata=dict(script=str(Path(__file__).relative_to(ROOT)),script_sha256=sha(Path(__file__)),inputs={str(p.relative_to(ROOT)):sha(p) for p in files},outputs={str((OUT/f'{n}.csv').relative_to(ROOT)):sha(OUT/f'{n}.csv') for n in outputs},environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),checks_passed=len(checks),common_bootstrap=dict(n=B,seed=seed,unit='household; same multiplicities across both clocks and benchmarks',projection='refitted within each resample; the specified household counts, control values and original winsorization cutoffs remain fixed'),jackknife='Delete each household jointly across endpoints; refit activity projection and renormalize weights; (G-1)/G times centered delete-one cross-products; Student t(G-1) interval is an approximate independent-household sensitivity.',interpolation='Y(lambda)=Y0+lambda*(Y1-Y0) for already transformed purchase outcomes. Not compounding fractional fitted alpha, and no new winsorization at interior lambda.',common_scope='Same 30 households; clock-specific purchase histories and cutoffs remain different. No pure timing interpretation.',restricted_inputs_available=False)
    (OUT/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(pd.DataFrame(jack).query('scale=="winsorized" and term=="difference"').to_string(index=False))
    print(pd.DataFrame(common).query('scale=="winsorized" and term in ["K11","K00","difference"]').to_string(index=False))
    print(pd.DataFrame(poly).to_string(index=False))
    print(f'Passed {len(checks)} baseline and identity comparisons.')

if __name__=='__main__':main()
