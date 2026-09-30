"""Uniform random full-remaining-pool baseline; append-only sibling run.

Uses the frozen run's environment, data cleaning, remaining-pool construction,
seed schedule and AUC function. Deliberately bypasses all model-based selection.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import runner


def simulate(replay, cfg, oracle, initial):
    selections, rounds, repeats = [], [], []
    for repeat in range(cfg.n_repeats):
        train = initial.copy()
        cumulative_nir = cumulative_farred = cumulative_dark = 0
        curve = []
        for rnd in range(1, cfg.n_rounds + 1):
            pool = replay.build_remaining_oracle_pool(oracle, train)
            assert len(pool) == len(oracle) - len(initial) - (rnd - 1) * cfg.recommend_k
            seed = cfg.seed_start + repeat * 1000 + rnd * 10 + 1
            indices = np.random.RandomState(seed).choice(len(pool), cfg.recommend_k, replace=False)
            chosen = pool.iloc[indices].copy()
            # Selection uses row indices only; labels are read after sampling.
            digest = __import__('hashlib').sha256('\n'.join(pool.Sequence).encode()).hexdigest()
            for rank, row in enumerate(chosen.itertuples(), 1):
                selections.append(dict(arm='random', repeat_id=repeat, round=rnd,
                    Rank=rank, Sequence=row.Sequence, class_label=row.class_label,
                    round_seed=seed, pool_order_sha256=digest))
            hits = int(chosen.class_label.eq('NIR').sum())
            farred = int(chosen.class_label.eq('Far Red').sum())
            dark = int(chosen.class_label.eq('Dark').sum())
            cumulative_nir += hits
            cumulative_farred += farred
            cumulative_dark += dark
            curve.append(cumulative_nir)
            rounds.append(dict(arm='random', repeat_id=repeat, round=rnd,
                round_seed=seed, batch_NIR=hits, cumulative_NIR=cumulative_nir,
                batch_FarRed=farred, batch_Dark=dark, batch_unsafe=farred+dark,
                cumulative_unsafe_rate=(cumulative_farred+cumulative_dark)/(rnd*cfg.recommend_k),
                candidate_pool_size=len(pool), pool_order_sha256=digest))
            train = pd.concat([train, chosen], ignore_index=True)
        repeats.append(dict(arm='random', repeat_id=repeat,
            AUC=replay.compute_learning_curve_auc(curve), final_NIR=cumulative_nir,
            final_FarRed=cumulative_farred, final_Dark=cumulative_dark,
            unsafe_rate=(cumulative_farred+cumulative_dark)/(cfg.n_rounds*cfg.recommend_k)))
    return pd.DataFrame(selections), pd.DataFrame(rounds), pd.DataFrame(repeats)


def validate(replay, cfg, oracle, initial, selected, rounds, repeated):
    assert len(selected) == 3600 and len(rounds) == 300 and len(repeated) == 30
    assert not selected.duplicated(['repeat_id', 'Sequence']).any()
    assert not selected.Sequence.isin(initial.Sequence).any()
    assert selected.class_label.tolist() == oracle.set_index('Sequence').loc[selected.Sequence].class_label.tolist()
    for repeat, records in selected.groupby('repeat_id'):
        measured = set(initial.Sequence)
        curve = []
        total = 0
        for rnd, batch in records.groupby('round'):
            # Independent reconstruction (not the helper used in simulate).
            pool = oracle.loc[~oracle.Sequence.isin(measured)].reset_index(drop=True)
            seed = 42 + repeat * 1000 + rnd * 10 + 1
            expected = pool.iloc[np.random.RandomState(seed).choice(len(pool), 12, replace=False)]
            assert batch.Sequence.tolist() == expected.Sequence.tolist()
            assert len(batch) == 12 and set(batch.Sequence).isdisjoint(measured)
            digest = __import__('hashlib').sha256('\n'.join(pool.Sequence).encode()).hexdigest()
            assert batch.pool_order_sha256.eq(digest).all()
            total += int(batch.class_label.eq('NIR').sum())
            curve.append(total)
            measured.update(batch.Sequence)
        actual = repeated.loc[repeated.repeat_id.eq(repeat)].iloc[0]
        assert actual.AUC == sum((curve[i]+curve[i+1])/2 for i in range(9))
        assert actual.final_NIR == total
    again = simulate(replay, cfg, oracle, initial)
    for a, b in zip((selected, rounds, repeated), again):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def main(source):
    source = source.resolve()
    replay, cfg, oracle, initial, *_ = runner.environment(source)
    assert (cfg.n_repeats, cfg.n_rounds, cfg.recommend_k, cfg.seed_start) == (30, 10, 12, 42)
    # Validate the four existing summaries before producing a combined plot table.
    main_metrics = pd.read_csv(source/'analysis/repeat_metrics.csv')
    old_summary = pd.read_csv(source/'analysis/performance_summary.csv').set_index('arm')
    assert len(main_metrics) == 120 and set(main_metrics.arm) == set(runner.ARMS)
    for arm, rows in main_metrics.groupby('arm'):
        assert np.isclose(rows.AUC.mean(), old_summary.loc[arm, 'AUC_mean'])
    out = source.parent / ('fig2_random_fullpool_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out/'random_baseline.py')
    design = dict(source_run=str(source), n_repeats=30, n_rounds=10, batch_size=12,
        initial_size=120, oracle_size=2962, initial_remaining_pool_size=2842,
        sampling='Uniform without replacement from all unmeasured oracle rows, original oracle order',
        RNG='numpy RandomState(seed).choice(pool_size, 12, replace=False)',
        seed_formula='42 + repeat_id*1000 + round*10 + 1',
        exclusions='Initial120 and previously measured sequences only',
        model_fitting=False, prior=False, SafeNIR=False, ABC=False, risk_filter=False,
        AUC='Trapezoidal integration of cumulative new NIR at rounds 1..10; no round 0',
        bootstrap='20000 shared seed-index resamples, RandomState(20260902), percentile 95% CI of mean',
        inference='Descriptive only; not added to the original primary hypothesis family',
        replication='30 algorithmic seeds on one fixed oracle; not biological replication')
    runner.save(out/'design.json', design)
    print(f'OUTPUT {out}', flush=True)
    selected, rounds, repeated = simulate(replay, cfg, oracle, initial)
    print('SIMULATED 30 repeats / 300 rounds / 3600 selections', flush=True)
    validate(replay, cfg, oracle, initial, selected, rounds, repeated)
    print('VALIDATED exact rerun, all selections, pool exclusions, labels and AUC', flush=True)
    selected.to_csv(out/'selected_sequences.csv', index=False)
    rounds.to_csv(out/'round_metrics.csv', index=False)
    repeated.to_csv(out/'repeat_metrics.csv', index=False)
    idx = np.random.RandomState(20260902).randint(30, size=(20000, 30))
    combined = pd.concat([main_metrics[['arm','repeat_id','AUC','final_NIR','unsafe_rate']],
        repeated[['arm','repeat_id','AUC','final_NIR','unsafe_rate']]], ignore_index=True)
    combined.to_csv(out/'five_arm_repeat_metrics.csv', index=False)
    summaries = []
    for arm in [*runner.ARMS, 'random']:
        values = combined.loc[combined.arm.eq(arm)].sort_values('repeat_id').AUC.to_numpy()
        lo, hi = np.quantile(values[idx].mean(1), [.025, .975])
        summaries.append(dict(arm=arm, mean=values.mean(), ci95_low=lo, ci95_high=hi,
            error_minus=values.mean()-lo, error_plus=hi-values.mean(), n_seeds=30))
    summary = pd.DataFrame(summaries).set_index('arm').loc[['full','no_prior','pnir','full_pool','random']].reset_index()
    summary.to_csv(out/'five_arm_auc_summary.csv', index=False)
    curve = rounds.pivot(index='repeat_id', columns='round', values='cumulative_NIR').sort_index().to_numpy()
    boot = curve[idx].mean(1)
    curve_table = pd.DataFrame(dict(arm='random', metric='cumulative_NIR', round=np.arange(1,11),
        mean=curve.mean(0), sd=curve.std(0, ddof=1), ci95_low=np.quantile(boot,.025,axis=0),
        ci95_high=np.quantile(boot,.975,axis=0), n_seeds=30))
    curve_table.to_csv(out/'random_learning_curve.csv', index=False)
    pd.concat([pd.read_csv(source/'analysis/learning_curve_data.csv'), curve_table], ignore_index=True).to_csv(
        out/'five_arm_learning_curve_data.csv', index=False)
    remaining = replay.build_remaining_oracle_pool(oracle, initial)
    nir_fraction = remaining.class_label.eq('NIR').mean()
    runner.save(out/'validation.json', dict(passed=True, frozen_source_environment_verified=True,
        exact_full_rerun=True, independently_reconstructed_300_pools=True,
        selections_validated=3600, repeat_AUC_validated=30,
        remaining_NIR_count=int(remaining.class_label.eq('NIR').sum()),
        analytical_random_expected_final_NIR=120*nir_fraction,
        analytical_random_expected_AUC=594*nir_fraction,
        random_mean_AUC=float(repeated.AUC.mean()), random_mean_final_NIR=float(repeated.final_NIR.mean()),
        numpy=np.__version__, pandas=pd.__version__))
    files = [p for p in out.iterdir() if p.is_file()]
    source_files = [source/'config.json', source/'provenance.json', source/'analysis/repeat_metrics.csv',
        source/'inputs/ALL2962.csv', source/'inputs/Initial120.csv',
        source/'inputs/offline_prior_replay_tripath_signed.py', Path(runner.__file__).resolve()]
    runner.save(out/'provenance.json', dict(created=datetime.now().isoformat(),
        output_sha256={p.name:runner.sha(p) for p in files},
        source_sha256={str(p):runner.sha(p) for p in source_files}))
    runner.save(out/'completion.json', dict(completed=True, repeats=30, rounds=300,
        selections=3600, validation_passed=True, source_run_unmodified=True))
    print(summary.to_string(index=False), flush=True)
    print('Random mean final NIR:', repeated.final_NIR.mean(), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    main(parser.parse_args().source_run)
