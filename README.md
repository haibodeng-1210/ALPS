# ALPS: Active Learning for Prior-guided Sequence Design

ALPS explores DNA sequence candidates for near-infrared-emitting DNA-stabilized silver nanoclusters through model-guided selection and experimental feedback.

This repository is a curated source collection: nine selected Python scripts plus this README. The scripts were copied without any content changes from `AL-DNA-active-learning_github_20260930.zip` and given descriptive English filenames. This collection is not a validated, immediately runnable reproduction package. Original import names, path settings, and historical model assumptions remain in the code.

## Selected files and purpose

| File | Purpose | Main inputs and outputs |
| --- | --- | --- |
| `alps_active_learning.py` | Main sequence-selection workflow: data cleaning, 144 staple descriptors, committee prediction, signed motif priors, local mutation / constrained de novo / random candidate generation, SafeNIR scoring, auxiliary wavelength and brightness regression, diversity, and greedy batch selection. | Training CSV and optional prior, exclusion, and gate files; recommendation tables, candidate tables, and run summaries. |
| `generate_signed_motif_priors.py` | Computes point-biserial correlations between staple features and NIR-versus-other labels; exports positive and negative motif rankings. | A labelled training CSV containing 144 descriptors; full signed ranking and positive/negative top-15 tables. |
| `replay_prior_guided_active_learning.py` | Retrospective oracle replay comparing signed-prior ON and OFF, with labels revealed after candidate selection and training data expanded each round; produces a prior-gate decision. | Historical oracle and starting training CSVs; selections, learning curves, repeated-run summaries, and gate JSON. |
| `analysis/fig2_main_elasticnet/run_elasticnet_ablation_replay.py` | Historical common-baseline replay comparing Full, No prior, P(NIR), and Full pool under paired random seeds, with input hashes and selection checkpoints. | Historical source snapshots, configuration files, oracle, Initial120, and signed priors; per-arm selections, candidate pools, metrics, and provenance. |
| `analysis/fig2_main_elasticnet/analyze_paired_ablation_results.py` | Reconstructs replay metrics from selected oracle labels, audits candidate pools, computes paired bootstrap intervals and sign-flip tests, and applies Holm correction. | Completed replay outputs and historical reference outputs; performance summaries, paired effects, learning curves, and validation records. |
| `analysis/fig2_main_elasticnet/run_random_sampling_baseline.py` | Adds a uniform random baseline using the same initial set, oracle, batch budget, and paired seed schedule; independently checks sample order and labels. | A completed historical four-arm replay; random selections and metrics plus combined five-arm plotting tables. |
| `analysis/figure3_origin/compute_committee_probability_shap.py` | Historical probability-space permutation SHAP analysis of the linear committee, including convergence checks, feature-rank stability, and descriptive observed NIR rates. | Cumulative training CSVs, a historical compatible committee implementation, and the final measured dataset; SHAP, importance, stability, and observed-rate tables. |
| `lii_normalization/extract_dna_staple_features.py` | Converts validated 10-mer DNA sequences into the standard 144 ordered base-pair separation counts, with the same descriptor naming order as the main workflow. | Sequence CSV; an augmented CSV containing the 144 staple descriptors. |
| `lii_normalization/fit_fluorescence_peaks.py` | Fits one to three Gaussian components to fluorescence spectra, selects automatic models using BIC or supplied peak counts, and exports peak positions and amplitudes. | Tecan-style spectrum CSVs; fitted-peak summaries, optional fit plots, and failed-fit records. |

These files prioritize the discovery workflow and the evidence supporting it. Byte-identical duplicates, exploratory initial-set/model screens, repeated historical drivers, and Origin-specific figure layout scripts were omitted. The descriptor extractor is retained because it gives a small, reusable entry point for preparing sequence inputs.

## Original-to-renamed mapping

All original paths below are relative to the uploaded package root, `AL-DNA-active-learning/`.

| Original path | Renamed path |
| --- | --- |
| `run_dna_agn_active_learning_tripath.py` | `alps_active_learning.py` |
| `nir_feature_importance_signed_analysis.py` | `generate_signed_motif_priors.py` |
| `offline_prior_replay_tripath_signed.py` | `replay_prior_guided_active_learning.py` |
| `analysis/fig2_main_elasticnet/runner.py` | `analysis/fig2_main_elasticnet/run_elasticnet_ablation_replay.py` |
| `analysis/fig2_main_elasticnet/analyze.py` | `analysis/fig2_main_elasticnet/analyze_paired_ablation_results.py` |
| `analysis/fig2_main_elasticnet/random_baseline.py` | `analysis/fig2_main_elasticnet/run_random_sampling_baseline.py` |
| `analysis/figure3_origin/prepare_data.py` | `analysis/figure3_origin/compute_committee_probability_shap.py` |
| `lii_normalization/Feature generator.py` | `lii_normalization/extract_dna_staple_features.py` |
| `lii_normalization/fit_lii_with_control.py` | `lii_normalization/fit_fluorescence_peaks.py` |

## Compatibility and reproduction notes

### Filename dependencies

The source collection preserves imports literally. The prior-generation and replay scripts dynamically load `run_dna_agn_active_learning_tripath.py` from the root. The SHAP script imports that module. The ablation driver copies both original root filenames into its frozen inputs, and the statistics and random-baseline scripts use `import runner` within their analysis directory. Renaming those files therefore breaks their existing lookups.

For execution with unchanged source bytes, the necessary original filenames can be restored in a separate working copy. The following optional shell commands undo the export renames; they do not change Python source contents. Run them from the collection root. Restoring names alone does not resolve the model-version mismatch described below.

```bash
mv alps_active_learning.py run_dna_agn_active_learning_tripath.py
mv generate_signed_motif_priors.py nir_feature_importance_signed_analysis.py
mv replay_prior_guided_active_learning.py offline_prior_replay_tripath_signed.py
mv analysis/fig2_main_elasticnet/run_elasticnet_ablation_replay.py analysis/fig2_main_elasticnet/runner.py
mv analysis/fig2_main_elasticnet/analyze_paired_ablation_results.py analysis/fig2_main_elasticnet/analyze.py
mv analysis/fig2_main_elasticnet/run_random_sampling_baseline.py analysis/fig2_main_elasticnet/random_baseline.py
mv analysis/figure3_origin/compute_committee_probability_shap.py analysis/figure3_origin/prepare_data.py
mv lii_normalization/extract_dna_staple_features.py 'lii_normalization/Feature generator.py'
mv lii_normalization/fit_fluorescence_peaks.py lii_normalization/fit_lii_with_control.py
```

### Mixed model versions in the supplied archive

The actual current main classifier is a **five-member, stratified-fold ANOVA-16 + shrinkage-LDA committee with inner-fold temperature calibration**. Its configuration uses `base_model_name="anova16_shrinkage_lda"`, `committee_size=5`, and `classifier_lda_shrinkage=0.50`. Auxiliary wavelength and brightness models still use ElasticNet regression; that does not make the classifier an Elastic Net classifier. The direct-execution recommendation batch is 14.

The replay scripts instead expect the historical **25-member Elastic Net logistic-regression committee**, and their default batch is 12. The root replay constructs `base_model_name="elasticnet_logreg"` and `committee_size=25`; the current main classifier rejects those settings. The ablation driver also calls the historical two-argument `build_base_estimator(cfg, random_state)` and sets logistic-regression parameters, whereas the uploaded current main function accepts only `cfg` and returns LDA. These are pre-existing compatibility problems, independent of the export renames.

The probability SHAP script allocates arrays for 25 linear members and reconstructs probabilities without the current LDA temperature calibration. Its assertion comparing reconstructed and model probabilities therefore cannot establish validity for the uploaded five-member calibrated model. It is retained as a valuable historical interpretation implementation, not as an attribution routine verified for the current LDA classifier.

The original archive README explicitly excludes historical input/source snapshots. Those compatible model snapshots would be needed to reproduce the historical Elastic Net replay and SHAP outputs with unchanged code. This export neither substitutes another model implementation nor claims reproduction of manuscript results.

### Data, paths, and environment

- The supplied ZIP includes no datasets, raw spectra, trained models, replay results, gate decisions, or historical configuration/provenance files. These inputs are also absent from this selection.
- Original relative paths and working-directory assumptions remain. Examples include `00_原始数据/`, `Iteration3_156.csv`, `Initial120.csv`, `全集重要性分析/`, and the spectrum-processing folders. Supply the corresponding inputs and use the appropriate original working directory before execution.
- The root replay defaults to `ALL2998.csv`, while the historical ablation workflow explicitly checks a 2,962-row `ALL2962.csv` oracle. The file name and dataset identity must be checked against the intended run.
- The main script selects candidates for one round. Laboratory measurements and insertion of verified results into the next training CSV remain external steps.
- The fluorescence fitter's original filename mentioned control-based LII fitting, but the retained implementation exports component peak positions and amplitudes. It does not perform control-based intensity normalization or establish cross-detector intensity comparability, so its new name describes peak fitting only.
- The historical paired statistics describe algorithmic seed variation on one retrospective oracle, not independent experimental replicates.

The original project dependency list was unpinned: `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `matplotlib`, and `openpyxl`. The selected probability-SHAP script additionally imports `threadpoolctl`. An installation starting point is:

```bash
pip install numpy pandas scipy scikit-learn joblib matplotlib openpyxl threadpoolctl
```

This is not a frozen environment. Historical replay checks require exact versions recorded in the corresponding saved provenance files, which are not included. Source syntax uses Python 3.10-compatible union type annotations.

## Scope of review

The uploaded archive contains 151 Python files, corresponding to 136 distinct byte contents after exact deduplication. All 151 were statically parsed to inventory functions, classes, and imports. The selected workflow, feature, replay, inference, statistical, and spectral-processing implementations were inspected for purpose and dependency assumptions. No selected source was edited, executed end to end, or repaired.

## Source identity verification

Every selected Python file is byte-for-byte identical to its source in `AL-DNA-active-learning_github_20260930.zip`. Only its filename was changed; the containing directory structure was retained. Comments, encodings, line endings, configuration values, and imports were preserved.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `alps_active_learning.py` | 94946 | `8913dc9a944fe4547fde02ef04d03d2a49e51422a693a600f9b21531aa3b8e38` |
| `generate_signed_motif_priors.py` | 8149 | `2adefa74746be2fbd939e99f40479775907ae22046971e914e5ff74b5db08d22` |
| `replay_prior_guided_active_learning.py` | 36480 | `ee56d411af2f86df54324b34a87860479aaea0dfc63d0ded77fabc4042018a5b` |
| `analysis/fig2_main_elasticnet/run_elasticnet_ablation_replay.py` | 14438 | `6b317bdec7d66fd5411addb81816013f424b83d5a453dcd270d43703fb15b370` |
| `analysis/fig2_main_elasticnet/analyze_paired_ablation_results.py` | 13184 | `f160ef9f57a59e987ed897884f9a3d5c6f32a9a2b14c471f449cf21dc6a48131` |
| `analysis/fig2_main_elasticnet/run_random_sampling_baseline.py` | 10093 | `422b09b2d231ab13be48de60b5ccd10cdcbc59e3860db42c06dd16e05ed5312e` |
| `analysis/figure3_origin/compute_committee_probability_shap.py` | 8098 | `79d89ba3ec8bfd743f1365a5575561b222df2b6126408ffd81ee24863eb7afc0` |
| `lii_normalization/extract_dna_staple_features.py` | 2716 | `345717e0b9efd1c0545de0930306a487954914e896f5b4b24c63c6fd5f7c6ec3` |
| `lii_normalization/fit_fluorescence_peaks.py` | 19790 | `50230389bdca22a33fcbd6f6f4827b11aaa9a84398a631792681237d225b3f13` |

All nine selected scripts passed Python syntax parsing. These checks establish preservation and syntax validity only; end-to-end model runs and manuscript result reproduction were not performed.
