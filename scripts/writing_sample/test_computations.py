"""Standard-library tests and command-line validation for the continuation calculations."""
from pathlib import Path
import sys,tempfile,subprocess,json,unittest,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
import restricted_consumers as rc
import merge_results


def purchases():
    rng=np.random.default_rng(7);n=240;blocks=np.repeat(np.arange(12),20);hh=np.tile(np.arange(20),12);x=rng.normal(0,40,n)
    d=pd.DataFrame(dict(sample='cohort',event_id=np.arange(n),household_index=hh,entry_block=blocks,period=np.where(blocks<6,'A','B'),alpha20_bp=x,x_low=-10000.,x_high=10000.,Y0_low=-10000.,Y0_high=10000.,Y1_low=-10000.,Y1_high=10000.,large_stock=(hh%3==0).astype(int)))
    noise=rng.normal(size=n);xd=x-pd.Series(x).groupby(blocks).transform('mean');noise-=np.asarray(xd)*(np.asarray(xd)@noise)/(np.asarray(xd)@np.asarray(xd))
    d['Y0_bp']=.25*x+blocks+noise;d['Y1_bp']=d.Y0_bp-x+.05*x
    return d


def securities():
    rows=[]
    for j,date in enumerate(pd.date_range('2015-01-02',periods=12,freq='28D')):
        for sec in range(40):
            x=(sec-20)*3.;session=1000+20*j
            rows.append(dict(sample='traded_securities',security_id=sec,formation_session=session,formation_date=date,fit_end_session=session-22,window_end_session=session+20,alpha20_bp=x,Y0_bp=2+.25*x,Y1_bp=2-.75*x,market_value=100+sec))
    return pd.DataFrame(rows)


def sales():
    rng=np.random.default_rng(9);n=12
    return pd.DataFrame(dict(clock='T',design='sales',scale='winsorized',household_index=np.arange(n),n_A=np.arange(n)+10,n_B=np.arange(n)+5,Y1_A=rng.normal(size=n),Y1_B=rng.normal(size=n),Y0_A=rng.normal(size=n),Y0_B=rng.normal(size=n)))


def mean_input():
    d=purchases();return d[['event_id','household_index','entry_block','Y0_bp','Y1_bp']]


def counts():return pd.DataFrame(dict(household_index=[1,2,3],purchase_signals=[20,10,10],sale_signals=[0,15,25]))


class Tests(unittest.TestCase):
    def test_known_FE_slope_and_compounding_identity(self):
        out=rc.purchase_continuation(purchases(),True)
        full=out[(out.subset=='all')&(out.regression=='Y0_on_x')];np.testing.assert_allclose(full.estimate,.25,atol=1e-12);np.testing.assert_allclose(full.event_study_slope,-.70,atol=1e-12);np.testing.assert_allclose(full.remainder_slope,.05,atol=1e-12);np.testing.assert_allclose(out.identity_gap.dropna(),0,atol=1e-12)
    def test_noisy_benchmark_predictor_is_calibrated_on_estimated_alpha(self):
        # Orthogonal signal and estimation error make attenuation exact.
        d=purchases();x=d.alpha20_bp.to_numpy();noise=np.sin(np.arange(len(d)));blocks=d.entry_block.to_numpy();noise-=pd.Series(noise).groupby(blocks).transform('mean').to_numpy();xc=x-pd.Series(x).groupby(blocks).transform('mean').to_numpy();noise-=xc*(xc@noise)/(xc@xc);noise*=np.sqrt((xc@xc)/(noise@noise));d['alpha20_bp']=x+noise;d['Y0_bp']=x;d['Y1_bp']=d.Y0_bp-d.alpha20_bp
        out=rc.purchase_continuation(d,True);np.testing.assert_allclose(out[(out.subset=='all')&(out.scale=='raw')&(out.regression=='Y0_on_x')].estimate,.5,atol=1e-12)
    def test_security_grid_and_known_date_slopes(self):
        r=rc.security_continuation(securities(),True);np.testing.assert_allclose(r.estimate,.25,atol=1e-12);self.assertTrue((r.formation_dates==12).all())
        d=securities();d.loc[d.formation_session==1020,'formation_session']=1021
        with self.assertRaises(ValueError):rc.security_continuation(d,True)
    def test_A4_common_constant_changes_mean_not_SE(self):
        d=mean_input();d.Y1_bp=d.Y0_bp+30;r=rc.mean_standard_error(d,True);np.testing.assert_allclose(r.se.iloc[0],r.se.iloc[1],atol=1e-12);self.assertAlmostEqual(r.estimate.iloc[1]-r.estimate.iloc[0],30)
    def test_A3_bilinear_identity(self):
        r=rc.sales_signed(sales(),True).iloc[0];self.assertAlmostEqual(r.K11,r.K00+r.KDD+r.K0D+r.KD0,places=12)
    def test_spread_limit_two_parameter_correction(self):
        r=rc.spread_limit(counts(),True).iloc[0];self.assertAlmostEqual(r.limit,4/3*np.sqrt(78/79),places=12)
    def test_empirical_baseline_guards(self):
        with self.assertRaises(ValueError):rc.purchase_continuation(purchases(),False)
        with self.assertRaises(ValueError):rc.mean_standard_error(mean_input(),False)
        with self.assertRaises(ValueError):rc.spread_limit(counts(),False)
    def test_frozen_A2_endpoints_and_A5_heuristic_width(self):
        d=pd.read_csv(ROOT/'data/v4/writing_sample/results.csv');a=d[(d.analysis=='A2')&(d.clock=='T')&(d.scale=='winsorized')&(d.term=='curve')]
        self.assertAlmostEqual(float(a[(a.statistic=='estimate')&(a.lambda_value==0)].value.iloc[0]),-828.8193092685,places=7)
        self.assertAlmostEqual(float(a[(a.statistic=='estimate')&(a.lambda_value==1)].value.iloc[0]),2263.58263848774,places=7)
        for (clock,term),g in d[(d.analysis=='A5')&(d.scale=='winsorized')].groupby(['clock','term']):
            z=g.set_index('statistic').value;self.assertLess(z.ci_low_effective_heuristic,z.ci_low_households);self.assertGreater(z.ci_high_effective_heuristic,z.ci_high_households)
    def test_merge_is_deterministic_and_guards_groups(self):
        with tempfile.TemporaryDirectory(prefix='writing_sample_merge_') as tmp:
            tmp=Path(tmp);comp=tmp/'components';comp.mkdir()
            cols=merge_results.RESULT_COLUMNS
            a=pd.DataFrame([dict(zip(cols,['X','T','raw','t','s',np.nan,1.5,'bp','i','src','X/1','','20','','','']))])
            b=pd.DataFrame([dict(zip(cols,['Y','T','raw','t','s',np.nan,2.5,'bp','i','src','Y/1','','20','','','']))])
            a.to_csv(comp/'a.csv',index=False);b.to_csv(comp/'b.csv',index=False)
            merge_results.merge(tmp/'results.csv',comp,require_all=False);h1=hashlib.sha256((tmp/'results.csv').read_bytes()).hexdigest()
            a.to_csv(comp/'a.csv',index=False);merge_results.merge(tmp/'results.csv',comp,require_all=False)
            self.assertEqual(h1,hashlib.sha256((tmp/'results.csv').read_bytes()).hexdigest())
            self.assertEqual(set(pd.read_csv(tmp/'results.csv').analysis),{'X','Y'})
            b.assign(result_key='X/1').to_csv(comp/'b.csv',index=False)
            with self.assertRaises(SystemExit):merge_results.merge(tmp/'results.csv',comp,require_all=False)
    def test_rebuild_is_idempotent(self):
        """Rebuilding after every consumer and computation module twice gives byte-identical results.csv with every group kept."""
        res=ROOT/'data/v4/writing_sample/results.csv'
        if not (ROOT/'data/v4/writing_sample/components/consumer_A1_purchases.csv').exists():self.skipTest('empirical components not built')
        groups_before=set(pd.read_csv(res).analysis)
        env=dict(__import__('os').environ,PYTHONPATH=str(ROOT/'src'),MPLBACKEND='Agg')
        digests=[]
        for _ in range(2):  # two consecutive rebuilds on this machine (last-digit floats can differ across machines)
            for cmd in ([sys.executable,str(ROOT/'scripts/writing_sample/compute_available.py')],[sys.executable,str(ROOT/'scripts/writing_sample/merge_results.py'),'--check']):
                p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env=env);self.assertEqual(p.returncode,0,p.stderr[-2000:])
            digests.append(hashlib.sha256(res.read_bytes()).hexdigest())
        self.assertEqual(digests[0],digests[1]);self.assertEqual(groups_before,set(pd.read_csv(res).analysis))
        for stem,groups in merge_results.registered_groups().items():self.assertTrue(set(groups)<=groups_before,stem)
    def test_all_CLI_consumers_write_single_results_file(self):
        records=[]
        with tempfile.TemporaryDirectory(prefix='writing_sample_tests_') as tmp:
            tmp=Path(tmp);target=tmp/'results.csv'
            for key,d in [('A1_purchases',purchases()),('A1_securities',securities()),('A3',sales()),('A4',mean_input()),('spread_limit',counts())]:
                source=tmp/f'{key}.csv';d.to_csv(source,index=False)
                cmd=[sys.executable,str(ROOT/'scripts/writing_sample/restricted_consumers.py'),key,'--input',str(source),'--results',str(target),'--synthetic-validation'];p=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);records.append(dict(consumer=key,exit_code=p.returncode,synthetic=True))
            self.assertEqual(set(pd.read_csv(target).analysis),{'A1_purchases','A1_securities','A3','A4','spread_limit'})
        (ROOT/'data/v4/writing_sample/consumer_validation.json').write_text(json.dumps(dict(empirical_inputs_used=False,CLI=records,consumer_sha256=hashlib.sha256((ROOT/'scripts/writing_sample/restricted_consumers.py').read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dependency_sha256=hashlib.sha256((ROOT/'scripts/coherence/restricted_checks.py').read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':unittest.main()
