"""Reconstruct metrics from selected truth; paired seed-level statistics only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import runner

METRICS = ['AUC', 'final_NIR', 'unsafe_rate', 'candidate_evaluations']
PRIMARY = {'no_prior': 'AUC', 'pnir': 'unsafe_rate', 'full_pool': 'AUC'}
NAMES = {'full':'完整模型', 'no_prior':'去除全部 prior', 'pnir':'P_NIR 替换 SafeNIR', 'full_pool':'全剩余候选池'}


def holm(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    corrected = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    result = np.empty(len(p)); result[order] = corrected
    return result


def effect(delta, seed, iterations=20000):
    rng = np.random.RandomState(seed)
    draws = delta[rng.randint(len(delta), size=(iterations, len(delta)))].mean(1)
    lower, upper = np.quantile(draws, [.025, .975])
    rng = np.random.RandomState(seed+100000)
    null = (rng.choice([-1.,1.], size=(iterations,len(delta))) * delta).mean(1)
    p = (1 + np.count_nonzero(np.abs(null) >= abs(delta.mean()) - 1e-15)) / (iterations+1)
    return float(delta.mean()), float(lower), float(upper), float(p)


def analyze(run):
    r, cfg, oracle, initial, _, _, _ = runner.environment(run)
    out = run / 'analysis'; out.mkdir(exist_ok=True)
    round_frames = []; repeat_rows = []; checks = []
    for arm in runner.ARMS:
        folder = run / 'results' / arm
        completed = json.loads((folder / 'completion.json').read_text(encoding='utf-8'))
        assert completed['repeats'] == 30 and completed['from_scratch'] and completed['prior_checkpoint_reuse'] == 0
        assert runner.sha(folder / 'selected_sequences.csv') == completed['selected_sha256']
        assert runner.sha(folder / 'round_metrics.csv') == completed['metrics_sha256']
        selected = pd.read_csv(folder / 'selected_sequences.csv')
        saved_rounds = pd.read_csv(folder / 'round_metrics.csv')
        assert len(selected) == 3600 and len(saved_rounds) == 300
        assert set(selected.repeat_id) == set(range(30))
        assert not selected.duplicated(['repeat_id','Sequence']).any()
        assert not selected.Sequence.isin(initial.Sequence).any()
        assert selected.class_label.tolist() == oracle.set_index('Sequence').loc[selected.Sequence].class_label.tolist()
        assert (selected.round_seed == 42 + selected.repeat_id*1000 + selected['round']*10 + 1).all()
        for repeat, s in selected.groupby('repeat_id', sort=True):
            grouped = s.groupby('round', sort=True)
            assert set(grouped.groups) == set(range(1,11)) and grouped.size().eq(12).all()
            hits = grouped.class_label.apply(lambda x: int(x.eq('NIR').sum()))
            unsafe = grouped.class_label.apply(lambda x: int(x.isin(['Far Red','Dark']).sum()))
            farred = grouped.class_label.apply(lambda x: int(x.eq('Far Red').sum()))
            dark = grouped.class_label.apply(lambda x: int(x.eq('Dark').sum()))
            rm = saved_rounds.loc[saved_rounds.repeat_id.eq(repeat)].sort_values('round')
            assert np.array_equal(rm.batch_nir_hits, hits)
            assert np.array_equal(rm.batch_farred_false_positive, farred)
            assert np.array_equal(rm.batch_dark_false_positive, dark)
            assert np.array_equal(rm.cumulative_new_nir_hits, hits.cumsum())
            measured = set(initial.Sequence)
            for rnd, batch in grouped:
                pool = pd.read_csv(folder / 'candidate_pools' / f'repeat_{repeat:02d}_round_{rnd:02d}.csv.gz')
                assert not pool.Sequence.duplicated().any() and not set(pool.Sequence) & measured
                assert set(batch.Sequence).issubset(set(pool.Sequence))
                digest = __import__('hashlib').sha256('\n'.join(pool.Sequence).encode()).hexdigest()
                assert batch.pool_order_sha256.eq(digest).all()
                assert len(pool) == int(rm.loc[rm['round'].eq(rnd), 'candidate_pool_size'].iloc[0])
                if arm == 'full_pool':
                    assert len(pool) == 2842 - (rnd-1)*12
                probs = pool[['P_Green','P_Red','P_FarRed','P_NIR','P_Dark']].to_numpy()
                assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
                assert np.allclose(probs.sum(1), 1, atol=2e-6)
                if arm == 'pnir':
                    assert np.allclose(pool.target_term, pool.P_NIR, atol=1e-7, rtol=0)
                else:
                    assert np.allclose(pool.target_term, r.al.compute_safe_nir_score(probs.astype(np.float32)), atol=1e-7, rtol=0)
                if arm == 'no_prior':
                    assert np.allclose(pool.motif_prior, 0)
                measured.update(batch.Sequence)
            curve = pd.DataFrame({'arm':arm, 'repeat_id':repeat, 'round':np.arange(1,11),
                'batch_NIR':hits.to_numpy(), 'cumulative_NIR':hits.cumsum().to_numpy(),
                'batch_unsafe':unsafe.to_numpy(), 'cumulative_unsafe_rate':unsafe.cumsum().to_numpy()/np.arange(12,121,12),
                'candidate_pool_size':rm.candidate_pool_size.to_numpy()})
            round_frames.append(curve)
            repeat_rows.append({'arm':arm,'repeat_id':repeat,
                'AUC':r.compute_learning_curve_auc(hits.cumsum().tolist()), 'final_NIR':int(hits.sum()),
                'unsafe_rate':float(unsafe.sum()/120), 'candidate_evaluations':int(rm.candidate_pool_size.sum())})
        checks.append({'arm':arm, 'selections':3600, 'pool_files':300, 'passed':True})
    rounds = pd.concat(round_frames,ignore_index=True)
    repeated = pd.DataFrame(repeat_rows)
    rounds.to_csv(out/'round_metrics_reconstructed.csv',index=False)
    repeated.to_csv(out/'repeat_metrics.csv',index=False)
    summary = repeated.groupby('arm')[METRICS].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'performance_summary.csv')
    full = repeated.loc[repeated.arm.eq('full')].set_index('repeat_id').sort_index()
    effects = []; differences = []; index = 0
    for arm in ['no_prior','pnir','full_pool']:
        other = repeated.loc[repeated.arm.eq(arm)].set_index('repeat_id').sort_index()
        for metric in METRICS:
            delta = (full[metric]-other[metric]).to_numpy(float)
            mean,lo,hi,p = effect(delta,20260902+index)
            effects.append({'control':arm,'metric':metric,'contrast':'full minus control','mean_difference':mean,
                'ci95_low':lo,'ci95_high':hi,'p_two_sided_signflip':p,'primary':PRIMARY[arm]==metric,'n_seed_pairs':30})
            differences.extend({'control':arm,'metric':metric,'repeat_id':j,'full_minus_control':v} for j,v in enumerate(delta))
            index += 1
    effects = pd.DataFrame(effects)
    for primary in [True,False]:
        mask = effects.primary.eq(primary)
        effects.loc[mask,'p_holm_within_family'] = holm(effects.loc[mask,'p_two_sided_signflip'])
    effects.to_csv(out/'paired_effects.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    curve_rows = []
    for arm, frame in rounds.groupby('arm'):
        for metric in ['cumulative_NIR','cumulative_unsafe_rate','candidate_pool_size']:
            matrix = frame.pivot(index='repeat_id',columns='round',values=metric).sort_index().to_numpy()
            idx = np.random.RandomState(20260902).randint(30,size=(20000,30))
            boot = matrix[idx].mean(1)
            for rnd in range(10):
                curve_rows.append({'arm':arm,'metric':metric,'round':rnd+1,'mean':matrix[:,rnd].mean(),
                     'sd':matrix[:,rnd].std(ddof=1),'ci95_low':np.quantile(boot[:,rnd],.025),
                     'ci95_high':np.quantile(boot[:,rnd],.975),'n_seeds':30})
    pd.DataFrame(curve_rows).to_csv(out/'learning_curve_data.csv',index=False)
    parity = []
    for arm, path in [('full',runner.POOL_REFERENCE/'abc_baseline/selected_sequences_by_round.csv'),
                      ('full_pool',runner.POOL_REFERENCE/'full_pool/prior_on/selected_sequences.csv')]:
        old = pd.read_csv(path); old = old.loc[old.arm.eq('tripath_signed_prior')].sort_values(['repeat_id','round','Rank'])
        new = pd.read_csv(run/'results'/arm/'selected_sequences.csv').sort_values(['repeat_id','round','Rank'])
        exact = old.Sequence.tolist() == new.Sequence.tolist()
        parity.append({'arm':arm, 'all_30_repeat_selection_order_exact':exact})
        assert exact, f'Unexpected deviation from validated Elastic Net {arm} reference'
    runner.save(out/'validation.json',{'passed':True,'arms':checks,'historical_elasticnet_parity':parity,
            'all_14400_selected_records_valid':True,'all_1200_candidate_pools_valid':True,
            'metrics_reconstructed_from_truth':True})
    lines = ['# Fig2：统一 Elastic Net 主流程的三项消融','',
        '四组全部从头运行：每组 30 个配对种子、每次 10 轮、每轮 12 条，共 14400 条选样记录。无旧检查点复用。',
        '完整模型是三个对比的同一基线。CPU Elastic Net：saga，C=0.3，l1_ratio=0.3，max_iter=6000；25 个委员会成员，144 维输入特征，80% 特征子采样。', '',
        '## 平均表现','', '|组别|发现曲线 AUC|最终新增 NIR|Far Red + Dark 比例|累计候选评分数|', '|---|---:|---:|---:|---:|']
    for arm in runner.ARMS:
        row = summary.loc[arm]
        lines.append(f'|{NAMES[arm]}|{row.AUC_mean:.3f}|{row.final_NIR_mean:.3f}|{row.unsafe_rate_mean*100:.2f}%|{row.candidate_evaluations_mean:.1f}|')
    lines += ['', '## 三个主要对比', '',
        '差值统一为完整模型减对照；AUC 越大越好，非目标比例越小越好。区间为逐项 95% 配对 percentile bootstrap CI，20,000 次重采样；双侧配对符号翻转检验 20,000 次，对三个主要检验作 Holm 校正。', '']
    for _, row in effects.loc[effects.primary].iterrows():
        scale = 100 if row.metric=='unsafe_rate' else 1
        unit = ' 个百分点' if scale==100 else ''
        lines.append(f'- {NAMES[row.control]}：{row.metric} 差值 {row.mean_difference*scale:+.3f}{unit}，95% CI [{row.ci95_low*scale:.3f}, {row.ci95_high*scale:.3f}]；原始 p={row.p_two_sided_signflip:.6g}，Holm p={row.p_holm_within_family:.6g}。')
    lines += ['', '## 与旧结果的区别及边界', '',
        '- prior/SafeNIR 不再使用旧 L2 实现；所有组使用相同配对种子公式。',
        '- SafeNIR 使用各组独立闭环 ABC 候选池，不再使用旧的跨组联合候选池。其新旧数值不能仅归因于正则化变更。',
        '- prior OFF 指移除全部 prior 信息；不等同于仅关闭 scoring prior 的 gate OFF。',
        '- ABC 消融为整体候选池策略对比，不是分别删除 A/B/C；本次主比较固定 scoring prior ON，不把旧 OFF 分层混入共同基线。',
        '- Wavelength/Brightness 辅助分支和最终风险过滤保留。',
        '- 使用主流程配套的 oracle 回放：A/B/C 容量 800/1200/300，不额外 prune；真实推荐从全序列空间生成候选，容量 20000/20000/5000 且启用 prune。此处不能写成真实实验流程逐项完全相同。',
        '- prior 固定为 Initial120 训练集导出，不从未选 oracle 的真实标签更新。',
        '- 30 个种子描述固定数据集上的算法随机性，不是 30 次生物实验；区间不是跨数据集泛化区间。',
        '- 三个主要对比和九个次要对比分别作 Holm 校正；效应区间仍为逐项区间，学习曲线区间为逐点区间，不是同时置信带。',
        '- 本次分析方案在新结果生成前固定，但已见过相关旧结果，不能称为预注册。',
        '- 不因均值方向不利或不显著而删种子、改终点或重选参数。', '',
        '## 验证', '',
        '- 启动前：重跑基线 repeat 0 的 10 轮，与历史 Elastic Net 选样及排名一致；原始串行主流程第一轮选样也一致。',
        '- 完成后：全部 1200 个候选池、14400 条选样记录及逐轮真值指标重建检查通过。',
        '- 完整模型与全池两组全部 30 次重复的选样序列、顺序与各自已有 Elastic Net 记录完全一致。',
        '- 输入、主流程、回放脚本、运行参数和环境版本已冻结；旧实验、正式 gate、Origin 及桌面/OneDrive 图件没有修改。', '',
        '详见 preflight.json、provenance.json、analysis/validation.json、analysis/paired_effects.csv。']
    (run/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    runner.save(out/'analysis_provenance.json',{'analysis_script':str(Path(__file__).resolve()),'sha256':runner.sha(__file__),
        'bootstrap_iterations':20000,'signflip_iterations':20000,'seed_base':20260902,
        'all_checks_passed':True})
    print(summary.to_string(), flush=True)
    print(effects.loc[effects.primary].to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run',type=Path)
    analyze(p.parse_args().run)
