"""Input guards and estimators shared by the consumers of the restricted exports
(scripts/writing_sample/restricted_consumers.py). Run alone, it applies one check to an exported input file.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t


def require(d,columns,unique=None):
    missing=set(columns)-set(d)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    if unique and d.duplicated(unique).any():raise ValueError(f'Duplicate keys: {unique}')
    return d


def finite(d,columns):
    if not np.isfinite(d[columns].to_numpy(float)).all():raise ValueError(f'Nonfinite values: {columns}')


def fit_scores(y,X,w,contrast):
    """Return WLS contrast and individual linearized score contributions."""
    y,X,w,contrast=map(lambda z:np.asarray(z,float),(y,X,w,contrast))
    if (w<=0).any() or not np.isfinite(np.r_[y,w,X.ravel(),contrast]).all():raise ValueError('Invalid WLS inputs')
    if np.linalg.matrix_rank(X*np.sqrt(w)[:,None])<X.shape[1]:raise ValueError('Rank-deficient design')
    bread=np.linalg.inv(X.T@(w[:,None]*X));beta=bread@(X.T@(w*y));e=y-X@beta
    scores=(X@(bread@contrast))*w*e
    return float(contrast@beta),scores


def cgm_scalar(scores,household,block,k):
    """Two-way CR1 scalar contrast, retaining its untruncated variance."""
    n=len(scores)
    if n<=k:raise ValueError('Need observations > regressors')
    cl=pd.DataFrame({'h':household,'b':block,'s':scores})
    parts=[];counts=[]
    for cols in [['h'],['b'],['h','b']]:
        sums=cl.groupby(cols,dropna=False).s.sum();G=len(sums)
        if G<2:raise ValueError('Need at least two clusters in each partition')
        parts.append(float(G/(G-1)*(n-1)/(n-k)*np.sum(sums*sums)));counts.append(G)
    raw=parts[0]+parts[1]-parts[2];df=min(counts[:2])-1
    se=np.sqrt(max(raw,0));return dict(variance_raw=raw,se=se,df=df,truncated=raw<0,household_clusters=counts[0],block_clusters=counts[1])


def interval(estimate,variance):
    if variance['se']<=0: return dict(ci_low=np.nan,ci_high=np.nan)
    width=t.ppf(.975,variance['df'])*variance['se']
    return dict(ci_low=estimate-width,ci_high=estimate+width)


def joint_difference(d):
    """Conditional two-way uncertainty of a paired covariance contrast.

    Period-A residuals, household variances and fitted return construction are
    fixed. Identical event keys must occur under both benchmark definitions.
    """
    cols=['clock','specification','household_index','event_id','entry_block','y_B_bp','x_A_bp','activity_log','event_weight','var_x_A_bp2']
    require(d,cols,['clock','specification','event_id']);finite(d,cols[5:])
    records=[]
    for clock,cd in d.groupby('clock'):
        if set(cd.specification)!={'principal','intercept_omitted'}:raise ValueError('Need both principal and intercept_omitted benchmarks')
        pairs={key:g.sort_values('event_id').reset_index(drop=True) for key,g in cd.groupby('specification')}
        a,b=pairs['principal'],pairs['intercept_omitted']
        for col in ['event_id','household_index','entry_block','activity_log','event_weight']:
            if not a[col].equals(b[col]):raise ValueError(f'Benchmark pair differs in {col}')
        vals=[];scores=[]
        for g in [a,b]:
            var=g.var_x_A_bp2.unique()
            if len(var)!=1 or var[0]<=0:raise ValueError('One positive period-A variance required per benchmark')
            X=np.column_stack([np.ones(len(g)),g.x_A_bp,g.activity_log])
            est,s=fit_scores(g.y_B_bp,X,g.event_weight,[0,var[0],0]);vals.append(est);scores.append(s)
        v=cgm_scalar(scores[0]-scores[1],a.household_index,a.entry_block,3);gap=vals[0]-vals[1]
        records.append(dict(clock=clock,estimate=gap,K11=vals[0],K00=vals[1],**v,**interval(gap,v)))
    return pd.DataFrame(records)


def midpoint_spreads(d):
    require(d,['record_id','signal_id','household_index','entry_block','buy','return_bp','amount_lower','amount_upper'],['record_id'])
    finite(d,['household_index','entry_block','buy','return_bp'])
    if not d.buy.isin([0,1]).all():raise ValueError('buy must equal 0 or 1')
    for col in ['household_index','entry_block','buy','return_bp']:
        if (d.groupby('signal_id')[col].nunique()!=1).any():raise ValueError(f'Conflicting signal-level {col}')
    lower=d.amount_lower.to_numpy(float);upper=d.amount_upper.to_numpy(float)
    known=np.isfinite(lower)&(lower>0)&((np.isfinite(upper)&(upper>=lower))|np.isposinf(upper))
    # A whole signal is excluded if ANY constituent record lacks an amount.
    d=d.copy();d['known']=known;d['finite_bracket']=known&np.isfinite(upper)
    summary=[]
    for policy in ['equal_all_signals','finite_midpoints','open_at_lower','open_at_1.5_lower','open_at_2_lower']:
        f=d.copy()
        if policy=='equal_all_signals':f['amount']=1.
        else:
            mask=f.groupby('signal_id')['finite_bracket' if policy=='finite_midpoints' else 'known'].transform('all');f=f[mask].copy()
            if f.empty:raise ValueError(f'No complete amount signals for {policy}')
            factor={'finite_midpoints':1,'open_at_lower':1,'open_at_1.5_lower':1.5,'open_at_2_lower':2}[policy]
            f['amount']=np.where(np.isfinite(f.amount_upper),(f.amount_lower+f.amount_upper)/2,factor*f.amount_lower)
        signals=f.groupby('signal_id',sort=True).agg(household_index=('household_index','first'),entry_block=('entry_block','first'),buy=('buy','first'),return_bp=('return_bp','first'),amount=('amount','sum'),records=('record_id','size'))
        for weighting in (['equal'] if policy=='equal_all_signals' else ['equal_matched','amount']):
            w=np.ones(len(signals)) if weighting!='amount' else signals.amount.to_numpy(float)
            X=np.column_stack([np.ones(len(signals)),signals.buy]);estimate,scores=fit_scores(signals.return_bp,X,w,[0,1]);v=cgm_scalar(scores,signals.household_index,signals.entry_block,2)
            summary.append(dict(policy=policy,weighting=weighting,signals=len(signals),purchase_signals=int(signals.buy.sum()),sale_signals=int((1-signals.buy).sum()),records=int(signals.records.sum()),spread_bp=estimate,**v,**interval(estimate,v)))
    return pd.DataFrame(summary)


def common_purchases(d):
    """Same eligible purchase episodes under both clocks and both benchmarks.

    Outcomes arrive separately winsorized at the frozen original clock-specific
    cutoffs. The common subset is selected before forming household means.
    """
    cols=['common_episode_id','household_index','period','security_id','trade_quarter','disclosure_quarter','Y1_T_bp','Y0_T_bp','Y1_P_bp','Y0_P_bp']
    require(d,cols,['common_episode_id']);finite(d,cols[-4:])
    if not d.period.isin(['A','B']).all():raise ValueError('period must equal A or B')
    for col in ['trade_quarter','disclosure_quarter']:
        q=pd.PeriodIndex(d[col].astype(str),freq='Q')
        year=q.year
        ok=np.where(d.period=='A',(year>=2015)&(year<=2020),(year>=2021)&(year<=2025))
        if not ok.all():raise ValueError('Both clocks must assign each episode to the same period')
    eligible=[]
    for h,g in d.groupby('household_index'):
        a=g[g.period=='A'];b=g[g.period=='B']
        ok=(len(a)>=10 and len(b)>=5 and a.security_id.nunique()>=3 and all(a[q].nunique()>=4 and b[q].nunique()>=2 for q in ['trade_quarter','disclosure_quarter']))
        if ok:eligible.append(h)
    f=d[d.household_index.isin(eligible)].copy();G=len(eligible)
    if G<4:raise ValueError('Fewer than four eligible households in exact common-purchase sample')
    rows=[]
    for clock in ['T','P']:
        for bench in ['Y1','Y0']:
            means=f.groupby(['household_index','period'])[f'{bench}_{clock}_bp'].mean().unstack('period').reindex(eligible)
            counts=f.groupby(['household_index','period']).size().unstack('period').reindex(eligible)
            w=np.sqrt(counts.A.to_numpy()*counts.B.to_numpy());c=np.log1p(counts.A.to_numpy());C=np.column_stack([np.ones(G),c]);X=means[['A','B']].to_numpy()
            if np.linalg.matrix_rank(C)<2:raise ValueError('Activity control rank deficient')
            R=X-C@np.linalg.lstsq(C*np.sqrt(w)[:,None],X*np.sqrt(w)[:,None],rcond=None)[0];k=np.sum(w*R[:,0]*R[:,1])/w.sum()
            rows.append(dict(clock=clock,benchmark=bench,households=G,purchases_A=int(counts.A.sum()),purchases_B=int(counts.B.sum()),covariance_bp2=k,slope=k/(np.sum(w*R[:,0]**2)/w.sum())))
    return pd.DataFrame(rows)


def hac_mean(values,lags=6):
    """Equal-date mean and Bartlett HAC standard error, preserving date order."""
    x=np.asarray(values,float);n=len(x)
    if n<max(12,lags+2):raise ValueError('Too few formation dates for HAC')
    e=x-x.mean();meat=e@e
    for lag in range(1,lags+1):meat+=2*(1-lag/(lags+1))*(e[lag:]@e[:-lag])
    return float(x.mean()),float(np.sqrt(max(meat,0))/n)


def alpha_forecast(d, evaluation_start="2015-01-01", evaluation_end="2025-12-31"):
    """Cross-sectional prediction of 20-session alpha-omitted returns.

    Predictor is 20 times the pre-formation daily fitted alpha; no winsorization. Equal weights
    within date and equal-date averaging. This estimates predictive calibration
    of noisy fitted alpha, not a structural latent-alpha persistence parameter.
    """
    cols=['security_id','formation_date','fit_end_date','window_end_date','shrcd','estimation_n','omitted_return_bp','fitted_return_bp','past_alpha20_bp']
    require(d,cols,['security_id','formation_date']);finite(d,['shrcd','estimation_n','omitted_return_bp','fitted_return_bp','past_alpha20_bp'])
    d=d.copy()
    for col in ['formation_date','fit_end_date','window_end_date']:d[col]=pd.to_datetime(d[col],errors='raise')
    if not (d.fit_end_date<d.formation_date).all() or not (d.window_end_date>d.formation_date).all():raise ValueError('Invalid chronology')
    if not d.shrcd.isin([10,11]).all() or (d.estimation_n<60).any():raise ValueError('Require CRSP common shares and >=60 fitting observations')
    rows=[]
    for date,g in d.groupby('formation_date',sort=True):
        X=np.column_stack([np.ones(len(g)),g.past_alpha20_bp]);b=np.linalg.lstsq(X,g.omitted_return_bp,rcond=None)[0]
        if len(g)<30 or np.linalg.matrix_rank(X)<2:raise ValueError('Need >=30 securities and variation in pre-formation fitted alpha at every formation date')
        rows.append(dict(formation_date=str(date.date()),window_end_date=str(g.window_end_date.max().date()),securities=len(g),intercept_bp=b[0],slope=b[1],mse_omitted=float(np.mean(g.omitted_return_bp**2)),mse_fitted=float(np.mean(g.fitted_return_bp**2))))
    dates=pd.DataFrame(rows);summ=[]
    dates['mse_fitted_minus_omitted']=dates.mse_fitted-dates.mse_omitted
    evaluation=dates[pd.to_datetime(dates.formation_date).between(pd.Timestamp(evaluation_start),pd.Timestamp(evaluation_end))]
    for variable in ['slope','intercept_bp','mse_omitted','mse_fitted','mse_fitted_minus_omitted']:
        est,se=hac_mean(evaluation[variable]);summ.append(dict(quantity=variable,estimate=est,hac_se=se,ci_low=est-t.ppf(.975,len(evaluation)-1)*se,ci_high=est+t.ppf(.975,len(evaluation)-1)*se,formation_dates=len(evaluation),hac_lags=6))
    # A benchmark learned using only completed prior return windows; never fit
    # on the next formation date or its realized future returns.
    forecasts=[]
    for date,g in d.groupby('formation_date',sort=True):
        if not pd.Timestamp(evaluation_start)<=date<=pd.Timestamp(evaluation_end):continue
        past=dates[(pd.to_datetime(dates.window_end_date)<date)]
        if len(past)<24:continue
        a=float(past.intercept_bp.mean());b=float(past.slope.mean());prediction=a+b*g.past_alpha20_bp
        forecasts.append(dict(formation_date=str(date.date()),training_dates=len(past),estimated_intercept_bp=a,estimated_slope=b,securities=len(g),mse_calibrated=float(np.mean((g.omitted_return_bp-prediction)**2)),mse_omitted=float(np.mean(g.omitted_return_bp**2)),mse_fitted=float(np.mean(g.fitted_return_bp**2))))
    forecasts=pd.DataFrame(forecasts)
    if len(forecasts)>=12:
        for comparator in ['omitted','fitted']:
            loss=forecasts.mse_calibrated-forecasts[f'mse_{comparator}'];est,se=hac_mean(loss)
            summ.append(dict(quantity=f'out_of_sample_mse_calibrated_minus_{comparator}',estimate=est,hac_se=se,ci_low=est-t.ppf(.975,len(forecasts)-1)*se,ci_high=est+t.ppf(.975,len(forecasts)-1)*se,formation_dates=len(forecasts),hac_lags=6))
    return dates,pd.DataFrame(summ),forecasts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('check',choices=['joint_difference','amount_weights','common_purchases','alpha_forecast']);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not args.input.is_file():p.error('Input is absent; no empirical result was computed')
    d=pd.read_csv(args.input);args.output.mkdir(parents=True,exist_ok=True)
    if args.check=='joint_difference':result={'joint_difference':joint_difference(d)}
    elif args.check=='amount_weights':result={'amount_spreads':midpoint_spreads(d)}
    elif args.check=='common_purchases':result={'common_purchase_statistics':common_purchases(d)}
    else:
        a,b,c=alpha_forecast(d);result={'alpha_date_regressions':a,'alpha_summary':b,'alpha_expanding_forecasts':c}
    for name,f in result.items():f.to_csv(args.output/f'{name}.csv',index=False,float_format='%.17g')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(dict(check=args.check,input=str(args.input),input_sha256=sha(args.input),script_sha256=sha(Path(__file__)),units='returns bp; covariance bp^2; amount USD',outputs={f'{n}.csv':sha(args.output/f'{n}.csv') for n in result}),indent=2)+'\n')

if __name__=='__main__':main()
