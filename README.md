# PolicyBench

How well can frontier models calculate tax and benefit outcomes without tools?

PolicyBench measures how well frontier AI models estimate selected household tax
and benefit outputs without tools.

For benchmark scope, snapshot policy, and terminology, see the
[benchmark card](docs/benchmark_card.md).

Benchmark scenarios are sampled from real households in the Enhanced CPS and then evaluated under 2025 policy rules with PolicyEngine-US. That gives the benchmark more realistic joint distributions of age, income, filing status, and family composition than independent synthetic sampling.

## Condition

1. **AI alone**: Models estimate tax/benefit values using only their training knowledge

## Benchmark scope

The default benchmark registry in code tracks the current published no-tools
leaderboard. Frozen paper claims are tied to
`results/paper_exports/benchmark_snapshot.json`, not to whatever models happen
to be configured or probed locally later.

## Programs evaluated

The current public release covers selected federal taxes, credits, means-tested
benefits, household eligibility labels, and state-tax outputs in the US, plus
selected tax and transfer outputs in the UK.

## Quick start

```bash
pip install -e ".[dev]"
pytest  # Run tests (mocked, no API calls)
```

## Benchmark run

```bash
# Generate reference outputs for 100 sampled CPS households
policybench ground-truth -n 100 --seed 42

# Run AI-alone evaluations on the exported scenario manifest
policybench eval-no-tools -n 100 --seed 42

# Analyze local results and export local artifacts
policybench analyze --output-dir results/local/analysis
```

## Repeated runs

```bash
# Optional: run the same benchmark multiple times on the saved scenario manifest
policybench eval-no-tools-repeated -n 100 --seed 42 --repeats 3 -o results/local/no_tools/runs

# Analyze the canonical point estimate plus across-run stability
policybench analyze --runs-dir results/local/no_tools/runs --output-dir results/local/analysis
```

`policybench ground-truth` writes `results/local/scenarios.csv`, and the eval
commands reuse that manifest by default instead of regenerating households from
the current source dataset. Prediction CSVs also get a `.meta.json` sidecar so
resumes only happen against the exact same manifest, model set, and program set.
