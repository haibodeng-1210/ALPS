"""Reproducible probability-space permutation SHAP for the actual AL committee.

No graphics are generated here. Origin owns all rendering.
"""
from pathlib import Path
import sys, json, hashlib, warnings
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import run_dna_agn_active_learning_tripath as al
warnings.filterwarnings('ignore',category=FutureWarning)
OUT=Path(__file__).resolve().parent/'data'
OUT.mkdir(parents=True,exist_ok=True)
FEATURES=al.STANDARD_FEATURES
FILES=['Initial120.csv','Iteration1_132.csv','Iteration2_144.csv','Iteration3_156.csv']

def save(df,name):
    df.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def main():
    frames=[pd.read_csv(ROOT/'00_原始数据'/f) for f in FILES]
    audit=[]
    for f,d in zip(FILES,frames):
        assert not d.Sequence.duplicated().any()
        assert np.isfinite(d[FEATURES].to_numpy(float)).all()
        audit.append(dict(file=f,n=len(d),n_nir=int(d.class_label.eq('NIR').sum()),sha256=hashlib.sha256((ROOT/'00_原始数据'/f).read_bytes()).hexdigest()))
    for a,b in zip(frames,frames[1:]):
        assert set(a.Sequence)<=set(b.Sequence)
        common=b.set_index('Sequence').loc[a.Sequence]
        np.testing.assert_array_equal(a[FEATURES].to_numpy(),common[FEATURES].to_numpy())
    # Fixed reference support, selected uniformly without using class labels.
    rng=np.random.default_rng(20260905)
    bg_idx=np.sort(rng.choice(len(frames[0]),64,replace=False))
    background=frames[0].iloc[bg_idx][FEATURES].to_numpy(float)
    save(frames[0].iloc[bg_idx][['Sequence']+FEATURES],'background64.csv')
    X=frames[-1][FEATURES].to_numpy(float)
    save(frames[-1][['Sequence','class_label']+FEATURES],'evaluation156.csv')
    path_background=np.tile(background,(4,1))
    orders=[rng.permutation(len(FEATURES)) for _ in path_background]
    all_summary=[]; checks=[]; full=[]
    for state,d in enumerate(frames):
        print(f'Training I{state+1}: n={len(d)}',flush=True)
        cfg=al.Config(n_jobs=4)
        committee=al.train_committee_model(d[FEATURES],d.class_label,cfg)
        W=np.zeros((25,5,144)); B=np.zeros((25,5))
        for m,member in enumerate(committee.models):
            sc=member.model.named_steps['scaler']; clf=member.model.named_steps['clf']
            for c,cls in enumerate(al.CLASS_ORDER):
                ci=list(clf.classes_).index(cls)
                coef=clf.coef_[ci]/sc.scale_
                W[m,c,member.selected_feature_indices]=coef
                B[m,c]=clf.intercept_[ci]-np.dot(coef,sc.mean_)
        def probs(x):
            return softmax(np.einsum('nf,mcf->nmc',x,W)+B,axis=2)[:,:,3]
        expected=committee.predict_proba_mean_std(X.astype(np.float32))[0][:,3]
        np.testing.assert_allclose(probs(X).mean(1),expected,atol=2e-7)
        acc=np.zeros((156,25,144)); half=None
        for k,(bg,order) in enumerate(zip(path_background,orders)):
            for perm in [order,order[::-1]]:
                logits=np.broadcast_to(np.einsum('f,mcf->mc',bg,W)+B,(156,25,5)).copy()
                old=softmax(logits,axis=2)[:,:,3]
                for j in perm:
                    logits+=(X[:,j]-bg[j])[:,None,None]*W[:,:,j][None,:,:]
                    new=softmax(logits,axis=2)[:,:,3]
                    acc[:,:,j]+=new-old
                    old=new
            if k==127: half=acc.mean(1)/256
            if (k+1)%64==0: print(f'I{state+1}: {k+1}/256 antithetic permutation pairs',flush=True)
        member_phi=acc/512
        phi=member_phi.mean(1)
        base=probs(background).mean()
        err=float(np.max(np.abs(base+phi.sum(1)-expected)))
        assert err<2e-7
        importance=np.abs(phi).mean(0)
        sd=np.abs(member_phi).mean(0).std(0,ddof=1)
        second=phi*2-half
        rho=float(spearmanr(np.abs(half).mean(0),np.abs(second).mean(0)).statistic)
        top_a=set(np.argsort(np.abs(half).mean(0))[-15:]);top_b=set(np.argsort(np.abs(second).mean(0))[-15:])
        checks.append(dict(state=state+1,n=len(d),background_n=64,paths=512,baseline_probability=float(base),max_additivity_error=err,split_half_spearman=rho,split_half_top15_jaccard=len(top_a&top_b)/len(top_a|top_b)))
        for j,f in enumerate(FEATURES):
            xv=d[f].to_numpy(); y=d.class_label.eq('NIR').to_numpy(float)
            r=float(np.corrcoef(xv,y)[0,1]) if np.ptp(xv)>0 else 0.
            direction=float(np.corrcoef(X[:,j],phi[:,j])[0,1]) if np.std(phi[:,j])>1e-12 and np.ptp(X[:,j]) else 0.
            all_summary.append(dict(state=state+1,training_size=len(d),feature=f,mean_abs_shap=importance[j],relative_importance_pct=100*importance[j]/importance.sum(),member_sd=sd[j],pearson_label=r,shap_value_correlation=direction))
        z=pd.DataFrame(phi,columns=FEATURES);z.insert(0,'Sequence',frames[-1].Sequence);z.insert(0,'state',state+1);full.append(z)
        np.savez_compressed(OUT/f'I{state+1}_member_shap.npz',shap=member_phi,features=FEATURES,sequences=frames[-1].Sequence.to_numpy(str))
        save(pd.DataFrame(all_summary),'importance_all.csv');save(pd.concat(full),'shap_all.csv');save(pd.DataFrame(checks),'convergence.csv')
    summary=pd.DataFrame(all_summary)
    top=summary[summary.state==4].nlargest(15,'mean_abs_shap').feature.tolist()
    pd.Series(top,name='feature').to_csv(OUT/'top15.csv',index=False)
    matrix=summary.pivot(index='feature',columns='state',values='relative_importance_pct').loc[top]
    matrix.reset_index().to_csv(OUT/'a_importance_heatmap.csv',index=False)
    ranks=summary.pivot(index='feature',columns='state',values='mean_abs_shap')
    stability=[]
    for i in range(1,5):
        for j in range(1,5):
            a=set(ranks[i].nlargest(15).index);b=set(ranks[j].nlargest(15).index)
            stability.append(dict(state_i=i,state_j=j,spearman=spearmanr(ranks[i],ranks[j]).statistic,top15_jaccard=len(a&b)/len(a|b)))
    save(pd.DataFrame(stability),'b_stability.csv')
    save(summary[summary.feature.isin(top[:6])],'c_trajectories.csv')
    # Every measured sequence included, initial and AL-acquired cohorts separate.
    final=pd.read_csv(ROOT/'00_原始数据'/'Iteration4_168.csv')
    assert not final.Sequence.duplicated().any()
    final['cohort']=np.where(final.Sequence.isin(frames[0].Sequence),'Initial','AL-acquired')
    save(final[['Sequence','class_label','cohort']+FEATURES],'observations168.csv')
    outcomes=[]
    for feature in top[:4]:
        for cohort,dd in final.groupby('cohort'):
            for count,g in dd.groupby(feature):
                n=len(g);k=int(g.class_label.eq('NIR').sum());p=k/n;z=1.95996398454
                center=(p+z*z/(2*n))/(1+z*z/n);radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                outcomes.append(dict(feature=feature,cohort=cohort,count=count,n=n,k=k,rate_pct=100*p,low_pct=100*(center-radius),high_pct=100*(center+radius)))
    save(pd.DataFrame(outcomes),'f_observed_rates.csv')
    (OUT/'audit.json').write_text(json.dumps(dict(inputs=audit,method='Monte Carlo interventional SHAP, NIR probability of actual 25-member elastic-net committee; 64 uniformly selected initial reference sequences, four permutations and their reverses per reference; fixed 156-sequence retrospective evaluation support',seed=20260905,top15=top,caveats=['Fixed evaluation support includes later-acquired sequences; interpretation is retrospective, not held-out predictive validation.','Independent feature masking can create off-manifold descriptor combinations.','Permutation SHAP is approximate; split-half convergence supplied.','Committee SD is member dispersion, not experimental uncertainty.','Observed rates are descriptive under adaptive selection, not independent validation.','No causal or nonlinear interaction claim inferred from attribution.']),indent=2),encoding='utf-8')
    print(pd.DataFrame(checks).to_string(index=False),flush=True)
    print('TOP:',top,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
