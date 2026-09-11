# 4K RULER starvation add-on

This add-on reuses the already-prepared 100-example 4K datasets and can run while the original FWE/NIAH/QA-1/QA-2 jobs are still running.

It does **not** rerun dense or any fixed S >= 8 method. Each task runs only:

```text
fixed1 fixed2 fixed4
adaptive0.5 adaptive1 adaptive2 adaptive4
```

Adaptive runs use the candidate set:

```text
1,2,4,8,16,32,64,128,256
```

This lowers the adaptive minimum from 8 to 1. The outputs are kept separate from the original sweep at:

```text
/shared/ruler/results/4096_100_starve/fwe
/shared/ruler/results/4096_100_starve/niah_multivalue
/shared/ruler/results/4096_100_starve/qa_1
/shared/ruler/results/4096_100_starve/qa_2
```

`summary.csv` now records `candidates`, `candidate_min`, and `candidate_max`, so the min-S=1 adaptive rows remain distinguishable from the original min-S=8 rows when results are concatenated later.

## Run from Windows

Commit and push the repo first. Existing running pods are unaffected because they already cloned their copy of the repo.

```bat
scripts\run-ruler-starvation100.bat
```

Then:

```bat
kubectl get jobs,pods -o wide
```

The script only deletes/recreates jobs whose names end in `starve100`; it does not touch the original jobs. Each add-on job checks that its existing dataset file is present before starting the model run.

The existing `scripts\download-ruler-results.bat` downloads the entire `/shared/ruler/results` tree, including both the original and starvation result directories.

## Old Tesla/Pascal GPU compatibility

The GPU jobs now require `nvidia.com/gpu.compute.major > 6` in addition to more than 20 GB of VRAM. This excludes Pascal-era GPUs such as Tesla P40 that cannot execute kernels from the current PyTorch/CUDA wheel, while retaining Volta/Turing/Ampere/Ada-class GPUs.

If only the FWE starvation job failed, rerun just that job without touching the others:

```bat
scripts\rerun-fwe-starvation100.bat
```
