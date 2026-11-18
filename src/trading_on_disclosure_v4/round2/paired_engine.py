#!/usr/bin/env python3
"""Exact paired household resampling from a COMPLETE sufficient-statistics export.

This program deliberately does not resample the incomplete public residual export.
Input columns: clock (T/P), scale (raw/winsorized), household_index, n_A, n_B,
Y1_A,Y1_B,Y0_A,Y0_B. Means must use the same purchases for both benchmarks.
For winsorized data, winsorize each purchase outcome at its original own cutoffs
BEFORE forming the supplied means. household_index must preserve the original
sorted household order to reuse the original random household selections.

The default seeds reproduce seed_from('tau/event_share',clock,'household bootstrap').
The same 4,999 multiplicity vectors are used for both benchmarks, every component,
and the raw/winsorized counterparts of each clock. Activity projections are refit.
No existing empirical file is modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
COLS=['clock','scale','household_index','n_A','n_B','Y1_A','Y1_B','Y0_A','Y0_B']
TERMS=['K11','K00','KDD','K0D','KD0','difference']


def original_seed(clock):
    return int(hashlib.sha256(f'tau/event_share|{clock}|household bootstrap'.encode()).hexdigest()[:8],16)


def evaluate(g, multiplicity=None):
    n_a=g.n_A.to_numpy(float); n_b=g.n_B.to_numpy(float)
    w=np.sqrt(n_a*n_b)
    if multiplicity is not None:
        w=w*np.asarray(multiplicity,float)
    keep=w>0
    if keep.sum()<3:
        return None
    C=np.column_stack([np.ones(keep.sum()),np.log1p(n_a[keep])])
    w=w[keep]; sw=np.sqrt(w); alpha=w/w.sum()
    A=g.loc[keep,['Y1_A','Y0_A']].to_numpy(float)
    B=g.loc[keep,['Y1_B','Y0_B']].to_numpy(float)
    # Each call re-estimates both activity projections on this resample.
    A-=C@np.linalg.lstsq(C*sw[:,None],A*sw[:,None],rcond=None)[0]
    B-=C@np.linalg.lstsq(C*sw[:,None],B*sw[:,None],rcond=None)[0]
    var_a=np.sum(alpha[:,None]*A*A,axis=0)
    var_b=np.sum(alpha[:,None]*B*B,axis=0)
    if np.any(var_a<=0):
        return None  # same degeneracy rule as the original bootstrap
    da=A[:,0]-A[:,1];db=B[:,0]-B[:,1]
    v={'K11':float(alpha@(A[:,0]*B[:,0])), 'K00':float(alpha@(A[:,1]*B[:,1])),
       'KDD':float(alpha@(da*db)), 'K0D':float(alpha@(A[:,1]*db)),
       'KD0':float(alpha@(da*B[:,1]))}
    v['difference']=v['K11']-v['K00']
    gap=v['K11']-sum(v[k] for k in ['K00','KDD','K0D','KD0'])
    assert abs(gap)<1e-8*max(1,abs(v['K11']))
    v.update(slope_Y1=v['K11']/var_a[0],slope_Y0=v['K00']/var_a[1],
             correlation_Y1=v['K11']/np.sqrt(var_a[0]*var_b[0]) if var_b[0]>0 else np.nan,
             correlation_Y0=v['K00']/np.sqrt(var_a[1]*var_b[1]) if var_b[1]>0 else np.nan,
             identity_gap=gap)
    return v


def run(path, output, n_boot=4999):
    path=Path(path);gall=pd.read_csv(path)
    missing=set(COLS)-set(gall)
    if missing:
        raise ValueError('Incomplete sufficient statistics; missing columns '+str(sorted(missing)))
    if set(zip(gall.clock,gall.scale))!={('T','raw'),('T','winsorized'),('P','raw'),('P','winsorized')}:
        raise ValueError('Input must contain both clocks and both scales.')
    if gall[COLS[2:]].isna().any().any() or not np.isfinite(gall[COLS[2:]].to_numpy(float)).all():
        raise ValueError('Input must be finite and complete.')
    if ((gall.n_A<1)|(gall.n_B<1)|(gall.n_A%1!=0)|(gall.n_B%1!=0)).any():
        raise ValueError('Actual positive integer purchase counts are required.')
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'data/v4/round2'):
        raise ValueError('Outputs must be under data/v4/round2; original data are frozen.')
    output.mkdir(parents=True,exist_ok=True)
    summary=[];draws=[];loo=[];points=[];metadata=[]
    for clock in ['T','P']:
        a=gall[(gall.clock==clock)&(gall.scale=='raw')].sort_values('household_index')
        b=gall[(gall.clock==clock)&(gall.scale=='winsorized')].sort_values('household_index')
        for field in ['household_index','n_A','n_B']:
            if not np.array_equal(a[field].to_numpy(),b[field].to_numpy()):
                raise ValueError(f'{clock}: raw and winsorized samples differ in {field}')
        if a.household_index.duplicated().any():
            raise ValueError('Household indices must be unique within each clock and scale.')
        n=len(a);seed=original_seed(clock);rng=np.random.default_rng(seed)
        multipliers=np.array([np.bincount(rng.integers(0,n,size=n),minlength=n) for _ in range(n_boot)])
        for scale,src in [('raw',a),('winsorized',b)]:
            g=src.reset_index(drop=True);point=evaluate(g)
            if point is None:raise ValueError('Degenerate original sample.')
            points.append({'clock':clock,'scale':scale,**point})
            boot=[]
            for i,m in enumerate(multipliers):
                v=evaluate(g,m)
                if v is not None:
                    boot.append(v);draws.append({'clock':clock,'scale':scale,'resample':i,**v})
            d=pd.DataFrame(boot)
            for term in TERMS:
                lo,hi=d[term].quantile([.025,.975])
                summary.append({'clock':clock,'scale':scale,'term':term,'estimate':point[term],
                                'bootstrap_se':d[term].std(ddof=1),'ci_low':lo,'ci_high':hi,
                                'n_boot':n_boot,'n_valid':len(d),'seed':seed})
            metadata.append({'clock':clock,'scale':scale,'seed':seed,'n_boot':n_boot,'n_valid':len(d),
                             'opposite_sign_share':float(((d.K11*d.K00)<0).mean()),
                             'directional_reversal_share':float(((d.K11>0)&(d.K00<0)).mean())})
            for j in range(n):
                m=np.ones(n);m[j]=0;v=evaluate(g,m)
                if v is None:raise ValueError('Degenerate leave-one-out sample.')
                w=np.sqrt(g.n_A.to_numpy(float)*g.n_B.to_numpy(float))
                loo.append({'clock':clock,'scale':scale,'omitted_household_index':int(g.household_index.iloc[j]),
                            'omitted_weight_share':float(w[j]/w.sum()),
                            **v,'K11_change':v['K11']-point['K11'],
                            'difference_change':v['difference']-point['difference']})
    for name, records in [('paired_summary',summary),('paired_resamples',draws),('point_statistics',points),
                          ('paired_signs',metadata),('leave_one_household_out',loo)]:
        pd.DataFrame(records).to_csv(output/(name+'.csv'),index=False)
    manifest={'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'n_boot':n_boot,'seed_convention':'sha256(tau/event_share|clock|household bootstrap) first 8 hex digits',
              'projection':'Refit [1,log(1+n_A)] using geometric weights times household multiplicity in every resample.',
              'cutoffs':'Held fixed; supplied winsorized means must already implement separately winsorized purchase outcomes.',
              'original_order_requirement':'household_index must retain original sorted household order to reproduce original random selections.',
              'precision':'All output values retained at machine precision; no reviewer rough value is used.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def self_check():
    sys.path.insert(0,str(ROOT/'src'))
    from trading_on_disclosure_v4.inference.block_bootstrap import _tau2_from_sums
    rng=np.random.default_rng(20261110);n=12
    g=pd.DataFrame({'n_A':rng.integers(10,250,n),'n_B':rng.integers(5,150,n),
                    'Y1_A':rng.normal(0,50,n),'Y1_B':rng.normal(0,50,n),
                    'Y0_A':rng.normal(0,50,n),'Y0_B':rng.normal(0,50,n)})
    w=np.sqrt(g.n_A*g.n_B).to_numpy(float);maxgap=0.
    for _ in range(128):
        m=np.bincount(rng.integers(0,n,n),minlength=n)
        ours=evaluate(g,m)
        for outcome in ['Y1','Y0']:
            value,slope=_tau2_from_sums(g.n_A.to_numpy(float),(g.n_A*g[outcome+'_A']).to_numpy(float),
                 g.n_B.to_numpy(float),(g.n_B*g[outcome+'_B']).to_numpy(float),w*m)
            maxgap=max(maxgap,abs(value-ours['K11' if outcome=='Y1' else 'K00']),
                       abs(slope-ours['slope_'+outcome]))
    assert maxgap<1e-9
    report={'kind':'synthetic_engine_validation_only','seed':20261110,'multiplicity_vectors':128,
            'original_implementation_comparisons':256,'max_abs_difference':maxgap,
            'passed':True,'empirical_paired_bootstrap_run':False}
    check=ROOT/'build/round2/paired_engine_self_check.json';check.parent.mkdir(parents=True,exist_ok=True)
    check.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'data/v4/round2/paired')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check()
    elif args.input is None:parser.error('--input requires the complete sufficient-statistics export; no approximation is allowed.')
    else:run(args.input,args.output)
