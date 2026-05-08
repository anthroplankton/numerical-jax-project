# AGENTS.md

Repository-wide instructions for Codex and AI-assisted contributors.

## Project goal

This repository is a course project for **Numerical Computation with JAX**.

The project has two core goals:

1. demonstrate numerical computation and training-oriented workflows with Python and JAX;
2. demonstrate how to run, monitor, and compare JAX workloads on **Google Cloud TPU**.

The final repository should support a clear presentation flow:

1. run small JAX examples locally;
2. run a small training-oriented JAX demo locally;
3. extend at least one demo to Google Cloud TPU;
4. collect logs, metrics, or screenshots that show what ran;
5. compare local and cloud or TPU execution using reproducible evidence;
6. summarize the workflow, results, limitations, and lessons learned for the course report.

This is not intended to become a production ML platform, production serving system, full MLOps pipeline, large-scale benchmark suite, or web application unless that scope is explicitly added later.

When goals conflict, prioritize:

1. runnable and understandable JAX examples;
2. Google Cloud TPU execution, monitoring, and cleanup workflows;
3. reproducible results and report-ready evidence;
4. local-versus-cloud comparison;
5. maintainable engineering practices;
6. Docker packaging or deployment support;
7. optional pretrained-model demonstrations.

## Repository layout

Prefer a simple Python project structure:

```text
README.md
AGENTS.md
pyproject.toml
uv.lock
src/
examples/
scripts/
tests/
docs/
cloud/
report/
configs/
```

Use these directories as follows:

- `src/`: reusable Python package code;
- `examples/`: runnable JAX demos and demo entry points;
- `scripts/`: diagnostics, setup helpers, result processing, and figure generation;
- `tests/`: pytest tests, smoke checks, and small fixtures;
- `docs/`: reusable technical documentation;
- `cloud/`: Google Cloud TPU setup, execution, monitoring, and cleanup workflows;
- `report/`: course-report materials, progress logs, figures, and curated results;
- `configs/`: example configuration files only, when repeated settings justify them.

Use `jax_tpu_project` as the default Python package name:

```text
src/jax_tpu_project/
```

Avoid creating multiple competing top-level packages. If the package name changes, update imports, examples, tests, and documentation consistently.

Use English names for files and directories. Use `snake_case.py` for Python files and modules. Use lowercase descriptive names for Markdown files and directories, such as `project_outline.md` or `tpu_workflow.md`. Avoid vague names such as `new_demo.py`, `final_v2.py`, `test_old.md`, or `copy.md`.

## Initial scaffolding order

Build the repository incrementally. Do not create every planned subsystem at once.

A suitable early order is:

1. repository skeleton and `pyproject.toml`;
2. local development tooling, such as `uv`, Ruff, pytest, and pre-commit;
3. local JAX sanity check that reports backend and devices;
4. basic JAX demo;
5. small training-oriented demo with smoke-test settings;
6. README, report templates, and progress log;
7. Google Cloud TPU documentation under `cloud/`;
8. TPU execution workflow once local demos are stable;
9. monitoring, result collection, and local-versus-cloud comparison;
10. Docker and CI extensions only when they support established workflows.

Do not add advanced cloud automation, pretrained-model demos, Docker deployment, or large CI workflows before the basic local JAX and training demos are working.

## Python, imports, and code style

Use the `src/` layout for reusable project code.

`examples/` and `scripts/` should be runnable entry points when practical. Move shared logic into `src/` instead of duplicating substantial logic across examples.

Avoid modifying `sys.path` inside examples or scripts unless there is a clear temporary reason. Prefer running examples and scripts from the repository root using documented `uv run ...` commands.

Follow PEP 8 conventions as enforced by Ruff. Use type hints for public functions and important internal functions when they improve readability.

Include module-level docstrings for runnable scripts, examples, and important modules. For runnable examples, briefly explain the purpose, expected execution context, and basic usage.

Prefer explicit, readable code that is easy to explain in reports, demos, and class presentations. Avoid overly compact implementations or premature abstractions that hide important JAX, training, or cloud workflow concepts.

For JAX and training code, make important assumptions explicit when relevant, including array shapes, dtype behavior, random seeds, backend assumptions, device assumptions, and metric definitions.

## Tooling and commands

Use `pyproject.toml` as the primary Python project configuration file.

When practical, keep project metadata, dependency groups, Ruff configuration, pytest configuration, and related Python tooling settings in `pyproject.toml`. Do not add `setup.cfg`, `tox.ini`, `requirements.txt`, `ruff.toml`, or other separate Python tooling configuration files unless they are explicitly needed.

Use `uv` for environment and dependency management. Do not manually edit lock files such as `uv.lock`; regenerate them with the appropriate package manager command when dependency changes require it.

For routine local validation, prefer these commands when the project environment is available:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Use automatic Ruff fixes or formatting only when appropriate for the current task. Do not reformat unrelated files or the entire repository unless formatting is the explicit task.

Pre-commit may be used for fast local checks, such as Ruff linting, Ruff formatting checks, trailing whitespace, end-of-file fixes, and YAML/TOML validation. Pre-commit hooks should not run heavy tests, training jobs, notebook execution, Docker builds, or Google Cloud commands.

CI should focus on local CPU validation: formatting checks, linting, unit tests, lightweight JAX import or device-detection checks, and fast demo smoke tests. CI should not require Google Cloud credentials, TPU quota, billing setup, cloud resource creation, pretrained-model access, or large downloads.

## Dependencies

Keep the default environment small enough to run basic JAX examples.

Basic examples should use raw JAX whenever practical. Training demos may use Optax, and may use Flax when it clearly improves structure or readability.

Use optional dependency groups for specialized workflows when practical:

- `dev`: linting, formatting, tests, pre-commit, and local development tools;
- `training`: Optax, Flax, metrics, or training-demo dependencies;
- `cloud`: Google Cloud, TPU, monitoring, or cloud-workflow dependencies;
- `pretrained`: Hugging Face, Gemma-like, or other pretrained-model demo dependencies.

Do not add, remove, upgrade, or downgrade dependencies unless the change is clearly needed for the current task.

When modifying dependencies, explain:

- why the change is needed;
- which workflow or demo uses it;
- whether it belongs in the default environment or an optional dependency group;
- whether setup documentation, CI, pre-commit, or lock files also need updates.

Do not add heavy ML, cloud, notebook, visualization, or pretrained-model dependencies to the default environment unless there is a clear project-level reason.

## Demos and acceptance criteria

Default demo settings should be small, quick, and safe to run. Basic examples, tests, and smoke checks should use CPU-friendly defaults whenever practical.

Longer, larger, cloud-specific, TPU-specific, Docker-specific, or pretrained-model runs should require explicit parameters or documented commands. Do not make expensive, long-running, or hardware-specific behavior the default execution path.

### Basic JAX demos

A basic JAX demo should run locally with small CPU-friendly defaults.

A complete basic demo should include, when practical:

- a documented command;
- clear expected output;
- a short explanation of the JAX concept being demonstrated;
- use of core JAX concepts such as JAX arrays, `jit`, `grad`, `vmap`, or backend/device awareness;
- simple checks or assertions when useful;
- no requirement for Google Cloud, TPU, large ML dependencies, pretrained models, or long-running computation.

### Training demos

A training-oriented demo should run end-to-end with small default or smoke-test settings.

A complete training demo should include, when practical:

- a documented command;
- CPU-friendly smoke-test settings;
- clear expected output;
- metrics such as loss, runtime, step time, throughput, and backend/device information;
- lightweight result artifacts, such as JSON, CSV, logs, or Markdown summaries;
- tests or smoke checks that verify the demo can run for a small number of steps;
- notes explaining what the demo demonstrates.

Training demos should not require long runs, large datasets, pretrained model weights, GPU, or TPU by default unless explicitly documented as advanced or hardware-specific workflows.

### Cloud TPU demos

Google Cloud TPU is a core learning target, not an optional afterthought.

A TPU-oriented demo should not be considered complete only because setup notes exist.

A complete TPU demo should include, when practical:

- documented setup prerequisites;
- the command or workflow used to run the JAX workload on TPU;
- clear expected output;
- captured logs, metrics, summaries, or screenshots that show the run occurred;
- monitoring or observability notes;
- cleanup instructions;
- local-versus-cloud comparison data or explanation;
- limitations, failed attempts, or unavailable resources when relevant.

If TPU execution is planned but not completed, mark it clearly as planned work rather than completed work.

### Pretrained-model demos

Pretrained-model demonstrations, including Gemma or similar open-weight models, are optional advanced demos.

Add them only when their scope and requirements are clearly understood. Document, when practical:

- model name and source;
- license, usage restrictions, or access requirements;
- required dependencies;
- expected hardware and approximate resource requirements;
- whether the demo is intended for local, Docker, GPU, TPU, or cloud execution;
- setup and run commands;
- expected output;
- limitations, failure modes, and fallback materials.

Pretrained-model demos should not be required for basic JAX examples, tests, CI, or default local setup.

## Script and example CLI conventions

Runnable scripts and examples should provide clear command-line interfaces when configuration is useful.

Use Python's standard `argparse` module unless a stronger CLI framework is clearly justified.

Expose important settings through command-line arguments when practical, such as:

- random seed;
- number of steps or epochs;
- batch size;
- learning rate;
- output directory;
- logging interval;
- demo scale;
- local, cloud, or TPU execution mode when relevant.

Runnable scripts should provide useful `--help` output. Avoid hiding important experiment settings in hardcoded constants when they affect reproducibility, metrics, result artifacts, or local-versus-cloud comparison.

## Testing

Use pytest for local tests and smoke checks.

Default tests should be lightweight, deterministic, and suitable for local CPU execution.

Tests should not require Google Cloud credentials, TPU quota, GPU, large datasets, downloaded model weights, pretrained-model access, or network access unless explicitly marked as optional or hardware-specific.

Use smoke tests for demos and training workflows. Run only a small number of steps and verify that expected metrics or outputs are produced.

Use standard pytest naming conventions, such as `test_*.py` files and `test_*` test functions. Test names should make the tested behavior clear, such as `test_jax_sanity_reports_backend` or `test_training_smoke_writes_metrics`.

Use `tests/fixtures/` for small fixtures when needed. Fixtures should be small, documented, and either synthetic or safe to redistribute.

Do not make long training runs, cloud workflows, Docker builds, or pretrained-model demos part of the default test suite.

If checks cannot be run because the environment, dependency, credential, quota, or cloud resource is unavailable, state that clearly instead of claiming they passed.

## Google Cloud safety and TPU workflow

Cloud and TPU-related instructions should be explicit, conservative, and clearly separated from local execution.

Do not assume that cloud credentials, project IDs, regions, zones, quotas, billing setup, TPU availability, or service accounts are already configured.

Commands that create, modify, reserve, start, stop, or delete Google Cloud resources should be treated carefully and should not be executed automatically.

Cloud documentation should state:

- where the command is run, such as local machine, Google Cloud Shell, or TPU VM;
- which project, region, zone, or resource it affects;
- whether it may incur cost;
- how to verify success;
- how to inspect active resources;
- how to clean up created resources.

Never commit credentials, service account keys, `.env` files, cloud authentication artifacts, downloaded model credentials, or project-specific secrets.

Use placeholders such as `<PROJECT_ID>`, `<REGION>`, `<ZONE>`, `<TPU_NAME>`, and `<BUCKET_NAME>` in documentation and example configs unless real values are intentionally safe to share.

Use Git as the primary way to move project code between local development and TPU VM environments. Avoid making untracked code edits directly on TPU VMs unless they are later brought back into the repository.

Cloud execution documentation should explain how the repository is obtained or updated on the TPU VM, how the environment is prepared, which command is run, where logs and outputs are written, how outputs are retrieved, and how resources are cleaned up.

## Monitoring, results, and reproducibility

Training and cloud workflows should include application-level metrics and infrastructure-level observations when applicable.

Record or report relevant information such as:

- command used;
- configuration values;
- Python and package versions when relevant;
- JAX backend and available devices;
- hardware or cloud environment;
- random seed;
- loss, accuracy, runtime, step time, throughput, and total runtime when applicable;
- TPU utilization, idle time, CPU usage, memory usage, network usage, logs, dashboards, or screenshots when available;
- output artifact paths.

Use fixed random seeds when practical. Do not assume bitwise-identical results across CPU, GPU, and TPU; document backend-related differences when they affect interpretation.

Use lightweight result artifacts such as JSON, CSV, Markdown summaries, or small text logs when they help reproducibility. Curated report artifacts may live under:

```text
report/results/
report/figures/
```

Do not commit large raw logs, model checkpoints, downloaded model weights, dataset caches, pretrained-model caches, or generated cloud artifacts unless they are intentionally reduced and documented.

When adding generated artifacts, document how they were produced and which script, command, or cloud workflow generated them.

Report-ready figures should have clear titles, labels, units, and captions when practical. Prefer generating figures from saved result artifacts so plots can be reproduced.

## Configuration files and secrets

Use example configuration files or documented placeholders for local and cloud settings.

Use `configs/` for example configuration files when repeated settings become useful:

```text
configs/
├── local.example.toml
├── training.example.toml
└── cloud.example.toml
```

Do not commit private local or cloud configuration files containing real project IDs, resource names, bucket names, credential paths, or other sensitive settings.

If private config files are used locally, make sure they are ignored by Git and documented through corresponding example files.

Keep `.gitignore` aligned with Python, JAX, training, cloud, and reporting workflows. Ignore common generated or private files, including Python caches, virtual environments, test artifacts, coverage artifacts, build artifacts, `.env` files, credentials, private configs, notebook checkpoints, raw logs, temporary outputs, generated experiment runs, model checkpoints, downloaded model weights, dataset caches, pretrained-model caches, and uncurated cloud artifacts.

Do not ignore important source files, example configs, documentation, or report materials by accident.

## Documentation and report materials

The README should be the main project entry point. It should include project purpose, repository layout, setup instructions, quick start commands, demo overview, links to docs, links to cloud workflows, links to report materials, current status, and known limitations when useful.

Use `docs/` for reusable technical explanations that would make the README too long.

Use `cloud/` for Google Cloud TPU operation documents, including setup, execution, monitoring, troubleshooting, and cleanup workflows.

Use `report/` for course-report materials and learning-oriented writeups:

```text
report/
├── final_report.md
├── project_outline.md
├── milestone_plan.md
├── progress_log.md
├── learning_notes.md
├── figures/
└── results/
```

Write code, comments, README files, cloud workflow documents, and reusable technical documentation in English.

Use Traditional Chinese for course-report materials, progress logs, learning reflections, and oral-presentation preparation when Chinese allows clearer explanation. Preserve important English technical terms, commands, library names, metrics, and cloud-service names where appropriate.

Report materials may be drafted with AI assistance, but they must remain grounded in actual project work. Do not fabricate results, logs, metrics, figures, screenshots, cloud runs, or completed work.

Course-report claims should be connected to runnable code, documented workflows, experiment results, logs, figures, screenshots, progress notes, or clearly identified external references.

When documenting JAX, Google Cloud, TPU, monitoring tools, libraries, pretrained models, or external datasets, prefer official documentation, primary sources, course materials, or clearly identified references. Do not invent citations, paper titles, documentation links, benchmark claims, model capabilities, or cloud-service behavior.

After meaningful project changes, update `report/progress_log.md` when it helps preserve what was done, what was learned, what issues appeared, and what should happen next.

When code changes affect usage, setup, dependencies, commands, outputs, or execution environments, update the relevant README, docs, cloud, or report materials.

## Docker and deployment

Docker may support reproducibility, packaging, or later Google Cloud deployment, but it is not the primary learning goal.

Keep direct Python and `uv run` workflows available for local development, learning, debugging, and reproducible demos whenever practical.

When Docker support is added, document what the image is for, how to build it locally, how to run it locally when practical, which Google Cloud service or workflow it supports, what credentials or project settings are required, and how deployment and cleanup are performed.

Docker CI jobs may verify that an image builds, but they should not push images, deploy services, create cloud resources, modify Google Cloud resources, or require Google Cloud credentials unless an explicit deployment workflow is added and reviewed.

## Notebooks

Notebooks may be used for exploration, teaching-oriented explanation, visualization, or presentation preparation.

Runnable Python scripts should remain the primary source for reproducible demos, tests, and cloud workflows.

When notebooks are added, keep core logic in reusable Python modules or scripts whenever practical. Avoid making notebooks the only way to run an important demo.

Clear notebook outputs before committing unless the outputs are intentionally small, relevant, and useful for the report.

## Agent working rules

Inspect existing files before editing.

Prefer small, focused, reviewable changes that directly address the current task.

Do not rewrite unrelated files, reformat the entire repository, introduce large new structures, or add new frameworks unless the task explicitly requires it.

Do not add new cloud automation, CI/CD configuration, Docker complexity, or pretrained-model workflows before the direct Python, JAX, and Google Cloud TPU workflows are clear.

Do not run destructive Git commands such as `git reset --hard`, `git clean -fd`, force pushes, or history rewrites unless explicitly requested.

Before deleting, moving, renaming, or rewriting many files, summarize the intended change and affected paths.

Be especially careful with report materials, cloud documentation, configuration files, generated result artifacts, figures, `LICENSE`, security-related files, and files that may contain project history or evidence.

Do not create Git commits unless explicitly requested. When useful, suggest a commit message following `<type>: <description>`.

Do not push changes to remote repositories unless explicitly requested. Before pushing, identify the target remote and branch, summarize the commits or changes, and report checks.

Do not claim that tests, cloud workflows, TPU runs, Docker builds, benchmarks, or pretrained-model demos passed unless they were actually run.

## Final response expectations

After making repository changes, summarize the work clearly.

A useful final summary should include:

- what changed;
- which files were modified or added;
- which commands or checks were run;
- which checks could not be run and why;
- any important limitations, assumptions, or follow-up steps.

For cloud, TPU, pretrained-model, Docker, or deployment-related work, explicitly state whether the change was documentation-only, locally tested, cloud-tested, or not tested.

## Nested AGENTS.md files

Start with this single root `AGENTS.md` for repository-wide instructions.

Add nested `AGENTS.md` files only when a subdirectory needs substantially different or more detailed instructions that would make the root file too long.

Good candidates for future nested instructions are:

- `cloud/AGENTS.md`, for detailed Google Cloud TPU safety, cost, execution, monitoring, and cleanup rules;
- `report/AGENTS.md`, for Traditional Chinese course-report style, academic-integrity rules, progress-log format, and evidence standards;
- `examples/AGENTS.md`, if demo-specific CLI, output, smoke-test, and artifact conventions become too detailed.

Do not duplicate shared instructions across multiple `AGENTS.md` files. Keep shared rules in the root file and use nested files only for directory-specific guidance.
