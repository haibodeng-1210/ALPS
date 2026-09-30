"""Frozen, common-baseline Elastic Net ablations; never edits historical runs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
import sklearn
import scipy

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('full', 'no_prior', 'pnir', 'full_pool')
REFERENCE = ROOT / 'analysis/startup_merged_elasticnet_20260902_183551'
POOL_REFERENCE = ROOT / 'analysis/initial120_abc_vs_fullpool_20260902_190916'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prepare():
    run = ROOT / 'analysis' / ('fig2_main_elasticnet_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    (run / 'inputs').mkdir(parents=True)
    (run / 'reference').mkdir()
    for name in ('run_dna_agn_active_learning_tripath.py', 'offline_prior_replay_tripath_signed.py'):
        shutil.copy2(ROOT / name, run / 'inputs' / name)
    for name in ('Initial120.csv', 'ALL2962.csv'):
        shutil.copy2(ROOT / '00_原始数据' / name, run / 'inputs' / name)
    shutil.copy2(REFERENCE / 'recomputed_importance/nir_feature_importance_signed_small.csv',
                 run / 'inputs/nir_feature_importance_signed_small.csv')
    shutil.copy2(REFERENCE / 'manifest.json', run / 'reference/main_manifest.json')
    shutil.copy2(REFERENCE / 'source_snapshot/run_dna_agn_active_learning_tripath.py',
                 run / 'reference/original_main.py')
    shutil.copy2(POOL_REFERENCE / 'config.json', run / 'reference/pool_config.json')
    shutil.copy2(Path(__file__), run / 'runner.py')
    base = json.loads((POOL_REFERENCE / 'config.json').read_text(encoding='utf-8'))
    base.update(oracle_csv=str(run / 'inputs/ALL2962.csv'), training_csv=str(run / 'inputs/Initial120.csv'),
                signed_prior_csv=str(run / 'inputs/nir_feature_importance_signed_small.csv'),
                output_root=str(run), use_gpu_backend=False, n_jobs=4)
    save(run / 'config.json', base)
    design = {
        'arms': list(ARMS), 'classifier_penalty': 'elasticnet',
        'main_reference': str(REFERENCE), 'repeats': 30, 'rounds': 10, 'batch': 12,
        'common_round_seed': '42 + repeat_id*1000 + round*10 + 1, identical for all four arms',
        'baseline': 'full ABC + signed prior + SafeNIR, fixed Initial120 prior',
        'no_prior': 'zero motif target; no prior in A/B construction; all 144 features for diversity',
        'pnir': 'replace compute_safe_nir_score with P_NIR only; independent training and ABC pool',
        'full_pool': 'replace ABC construction with all unmeasured oracle rows, original oracle order',
        'unchanged': 'classification/auxiliary models, class balancing, feature subsampling, acquisition weights, greedy selection and final risk filters',
        'offline_boundary': 'main companion oracle replay: A/B/C=800/1200/300, no candidate pruning; not full-space prospective candidate generation',
        'old_safe_protocol_not_reused': 'No coupled union candidate pool; each arm uses only its own feedback',
        'uncertainty_unit': 'algorithmic seed on one fixed retrospective oracle, not biological replicate',
        'primary_endpoints': {'prior': 'AUC', 'SafeNIR': 'unsafe_rate', 'candidate_pool': 'AUC'},
        'statistics': '20000 paired bootstrap draws and 20000 two-sided sign flips; Holm correction over 3 primary tests; secondary tests Holm over 9 arm-by-secondary comparisons; pointwise percentile CIs',
        'analysis_status': 'analysis plan fixed before new outputs; informed by previous related results, not preregistered',
        'all_arms_fresh': True,
    }
    save(run / 'design.json', design)
    paths = [*list((run / 'inputs').glob('*')), *list((run / 'reference').glob('*')),
             run / 'runner.py', run / 'config.json', run / 'design.json']
    hashes = {str(p.relative_to(run)): sha(p) for p in paths if p.is_file()}
    save(run / 'provenance.json', {'sha256': hashes, 'python': sys.version, 'numpy': np.__version__,
         'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'scipy': scipy.__version__,
         'created': datetime.now().isoformat()})
    print(run, flush=True)


def environment(run):
    provenance = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
    for name, digest in provenance['sha256'].items():
        assert sha(run / name) == digest, f'Frozen file changed: {name}'
    for name, value in [('python', sys.version), ('numpy', np.__version__), ('pandas', pd.__version__),
                        ('sklearn', sklearn.__version__), ('scipy', scipy.__version__)]:
        assert provenance[name] == value, f'Environment changed: {name}'
    r = load('fig2_frozen_replay', run / 'inputs/offline_prior_replay_tripath_signed.py')
    original = r.al.build_base_estimator
    def elasticnet(cfg, random_state):
        return original(cfg, random_state).set_params(clf__penalty='elasticnet')
    r.al.build_base_estimator = elasticnet
    cfg = r.ReplayConfig(**json.loads((run / 'config.json').read_text(encoding='utf-8')))
    oracle = r.add_digits_column(r.al.load_and_clean_data(cfg.oracle_csv))
    initial = r.add_digits_column(r.al.load_and_clean_data(cfg.training_csv))
    assert len(oracle) == 2962 and len(initial) == 120
    assert not oracle.Sequence.duplicated().any()
    assert set(initial.Sequence).issubset(set(oracle.Sequence))
    assert oracle.Sequence.str.fullmatch('[ACGT]{10}').all()
    a = initial.set_index('Sequence').sort_index()
    b = oracle.set_index('Sequence').loc[a.index]
    assert a.class_label.equals(b.class_label)
    assert np.allclose(a[r.al.STANDARD_FEATURES], b[r.al.STANDARD_FEATURES])
    pos, neg, prior = r.al.load_signed_motif_prior_from_csv(cfg.signed_prior_csv, cfg.motif_top_k)
    features = prior.feature.tolist()
    nir = initial.class_label.eq('NIR')
    assert np.allclose(initial.loc[nir, features].mean(), prior.mean_in_NIR, atol=1e-12, rtol=0)
    assert np.allclose(initial.loc[~nir, features].mean(), prior.mean_in_nonNIR, atol=1e-12, rtol=0)
    runtime = r.build_runtime_cfg(cfg, True)
    clf = r.al.build_base_estimator(runtime, 53).named_steps['clf']
    assert (clf.penalty, clf.solver, clf.C, clf.l1_ratio, clf.max_iter) == ('elasticnet', 'saga', .3, .3, 6000)
    assert not cfg.use_gpu_backend
    warnings.filterwarnings('ignore', category=FutureWarning)
    return r, cfg, oracle, initial, pos, neg, prior


def no_prior_pool(r, oracle, train, cfg, runtime):
    """Preserve the previous complete-prior ablation's explicit null design."""
    al = r.al
    remaining = r.build_remaining_oracle_pool(oracle, train)
    seeds = train.loc[train.class_label.isin(['NIR', 'Far Red'] if cfg.path_a_include_farred_seeds else ['NIR'])]
    if seeds.empty:
        seeds = train.loc[train.class_label.ne('Dark')]
    dist = al.compute_min_hamming_to_reference(np.asarray(remaining._digits.tolist(), dtype=np.uint8),
                                               np.asarray(seeds._digits.tolist(), dtype=np.uint8))
    path_a = remaining.loc[dist <= cfg.path_a_hamming_radius].copy()
    path_a['_pathA_distance'] = dist[dist <= cfg.path_a_hamming_radius]
    if len(path_a) > cfg.path_a_pool_size:
        rng = np.random.RandomState(runtime.random_seed + 1101)
        path_a = path_a.iloc[rng.choice(np.arange(len(path_a)), cfg.path_a_pool_size, replace=False)]
    path_a = path_a.sort_values('Sequence').reset_index(drop=True)
    path_a['candidate_path'] = 'A_local_oracle_no_prior'
    work = remaining.loc[~remaining.Sequence.isin(path_a.Sequence)].copy()
    work = work.loc[work.Sequence.map(lambda s: sum(ch in {'C', 'G'} for ch in s) >= cfg.path_b_min_cg_count)].copy()
    if not work.empty:
        features = al.STANDARD_FEATURES
        work['pathB_novelty_only'] = al.normalized_l1_distance_to_reference(
            work[features].to_numpy(np.float32), train[features].to_numpy(np.float32),
            al.build_feature_theoretical_max_vector(features))
        path_b = work.sort_values(['pathB_novelty_only', 'Sequence'], ascending=[False, True]).head(cfg.path_b_pool_size).copy()
        path_b['candidate_path'] = 'B_explore_oracle_no_prior'
    else:
        path_b = work
    path_c = r.build_oracle_path_C_pool(remaining, path_a, path_b, cfg, runtime.random_seed) if cfg.path_c_enabled else pd.DataFrame()
    if not path_c.empty:
        path_c['candidate_path'] = 'C_random_oracle_no_prior'
    return pd.concat([x for x in [path_a, path_b, path_c] if not x.empty], ignore_index=True).drop_duplicates('Sequence').reset_index(drop=True)


def configure_arm(r, arm):
    base_runtime = r.build_runtime_cfg
    base_pool = r.build_oracle_tripath_pool
    if arm == 'no_prior':
        def runtime(cfg, use_motif_prior):
            out = base_runtime(cfg, False)
            out.w_motif = 0.0
            out.staple_diversity_top_k = len(r.al.STANDARD_FEATURES)
            return out
        r.build_runtime_cfg = runtime
        def pool(oracle_df, train_df, cfg, pos_weights, neg_weights, full_signed_prior_df, runtime_cfg):
            return no_prior_pool(r, oracle_df, train_df, cfg, runtime_cfg)
        r.build_oracle_tripath_pool = pool
    elif arm == 'pnir':
        r.al.compute_safe_nir_score = lambda mean_probs: mean_probs[:, r.al.CLASS_ORDER.index('NIR')].astype(np.float32)
    elif arm == 'full_pool':
        def pool(oracle_df, train_df, cfg, pos_weights, neg_weights, full_signed_prior_df, runtime_cfg):
            out = r.build_remaining_oracle_pool(oracle_df, train_df)
            out['candidate_path'] = 'all_remaining_oracle'
            return out
        r.build_oracle_tripath_pool = pool
    return base_runtime, base_pool


def run_arm(run, arm, limit=30):
    r, cfg, oracle, initial, pos, neg, prior = environment(run)
    configure_arm(r, arm)
    if arm == 'no_prior':
        pos, neg, prior = {}, {}, pd.DataFrame()
    out = run / ('smoke' if limit != 30 else 'results') / arm
    out.mkdir(parents=True, exist_ok=False)
    (out / 'candidate_pools').mkdir()
    (out / 'checkpoints').mkdir()
    save(out / 'runtime_config.json', asdict(r.build_runtime_cfg(cfg, arm != 'no_prior')))
    all_metrics, all_selected = [], []
    started = time.time()
    base_screen = r.al.screen_candidate_space
    current = {}
    def screen(*args, **kwargs):
        screened = base_screen(*args, **kwargs)
        pool = screened['candidate_df']
        train = kwargs['train_df']
        step = (len(train) - 120) // 12 + 1
        assert not pool.Sequence.duplicated().any() and not pool.Sequence.isin(train.Sequence).any()
        if arm == 'full_pool':
            assert len(pool) == 2842 - 12 * (step - 1)
        table = pool[['Sequence', 'candidate_path']].copy()
        for idx, label in enumerate(r.al.CLASS_ORDER):
            table['P_' + label.replace(' ', '')] = screened['mean_probs'][:, idx]
        table['target_term'] = screened['safe_nir_score']
        table['motif_prior'] = screened['motif_prior_signed']
        if arm == 'no_prior':
            assert np.allclose(table.motif_prior, 0)
        digest = hashlib.sha256('\n'.join(pool.Sequence).encode()).hexdigest()
        current.setdefault('hashes', {})[step] = digest
        table.to_csv(out / 'candidate_pools' / f'repeat_{current["repeat"]:02d}_round_{step:02d}.csv.gz', index=False)
        return screened
    r.al.screen_candidate_space = screen
    for repeat in range(limit):
        current['repeat'] = repeat
        print(f'START {arm} repeat={repeat+1}/{limit}', flush=True)
        t0 = time.time()
        # Use the reference ON arm name for every cell to retain identical random streams.
        metrics, selected = r.run_single_arm_replay(oracle, initial, cfg, repeat,
                            'tripath_signed_prior', arm != 'no_prior', pos, neg, prior)
        for df in (metrics, selected):
            df['arm'] = arm
            df['round_seed'] = 42 + repeat * 1000 + df['round'] * 10 + 1
            df['pool_order_sha256'] = df['round'].map(current['hashes'])
        assert len(selected) == 120 and not selected.Sequence.duplicated().any()
        assert not selected.Sequence.isin(initial.Sequence).any()
        assert selected.groupby('round').size().eq(12).all()
        assert selected.class_label.tolist() == oracle.set_index('Sequence').loc[selected.Sequence].class_label.tolist()
        metrics.to_csv(out / 'checkpoints' / f'repeat_{repeat:02d}_metrics.csv', index=False)
        selected.to_csv(out / 'checkpoints' / f'repeat_{repeat:02d}_selected.csv', index=False)
        all_metrics.append(metrics); all_selected.append(selected)
        save(out / 'progress.json', {'arm': arm, 'completed_repeats': repeat+1, 'total': limit,
             'elapsed_seconds': time.time()-started})
        print(f'DONE {arm} repeat={repeat+1}/{limit} seconds={time.time()-t0:.1f}', flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    metrics.to_csv(out / 'round_metrics.csv', index=False)
    selected.to_csv(out / 'selected_sequences.csv', index=False)
    save(out / 'completion.json', {'arm': arm, 'repeats': limit, 'rounds': len(metrics),
         'selections': len(selected), 'elapsed_seconds': time.time()-started,
         'from_scratch': True, 'prior_checkpoint_reuse': 0,
         'selected_sha256': sha(out / 'selected_sequences.csv'), 'metrics_sha256': sha(out / 'round_metrics.csv')})
    print(f'COMPLETE {arm}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run', 'smoke'])
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--arm', choices=ARMS)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare()
    else:
        assert a.run_dir is not None and a.arm is not None
        run_arm(a.run_dir, a.arm, 1 if a.mode == 'smoke' else 30)
