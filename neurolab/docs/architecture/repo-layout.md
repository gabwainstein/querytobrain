# NeuroLab Repo Layout (Target)

NeuroLab lives as a **subproject** under `querytobrain`. Code may live in `querytobrain` or a sibling package; this layout describes the **documentation and intended code layout** for NeuroLab.

## Current layout (querytobrain)

```
querytobrain/
├── docs/
│   └── external-specs/
│       ├── NeuroLab_Plugin_Spec_v0.3.md   # Canonical product + eng spec
│       └── neurolab_implementation_guide.md        # Dataset catalog, registry, roadmap
└── neurolab/                                        # NeuroLab subproject
    ├── README.md
    ├── enrichment/              # CognitiveDecoder, UnifiedEnrichment, receptor, neuromaps, text_to_brain, scope_guard
    ├── scripts/                # Build caches, verify_*, query, predict_map, pipeline, etc.
    ├── data/                   # decoder_cache, neuromaps_cache, etc. (gitignored)
    ├── requirements-enrichment.txt
    └── docs/
        ├── README.md            # Docs map
        ├── product/             # Product truth
        ├── architecture/        # Baseline + ADRs
        ├── implementation/     # Slices, runbooks, REPO_REVIEW.md
        └── process/            # DoR, DoD, documentation rules, issue types
```

## Intended layout (when plugin code exists)

- **Plugin package:** Either under `querytobrain/packages/neurolab-plugin/` (or similar) or in a sibling repo. Must expose the tool contract in [interfaces.md](interfaces.md).
- **Dataset registry:** Config or DB under plugin or a dedicated `neurolab/data-registry/` (schema in implementation guide and [Schema Index](../implementation/SCHEMA_INDEX.md)).
- **Viewer:** Can live in `querytobrain/neurolab/viewer/` or separate repo; consumes manifest only.
- **Data services:** License-separated stores; implementation may use local disk, S3-compatible, or DB; see implementation specs.

## Conventions

- **Docs:** All normative product/architecture/process docs live under `neurolab/docs/`. Canonical long-form specs remain in `docs/external-specs/` and are linked from `neurolab/docs/`.
- **ADRs:** Under `neurolab/docs/architecture/adrs/`; numbered ADR-0001, ADR-0002, …
- **Diagrams:** `neurolab/docs/architecture/diagrams/` (e.g. `.mmd` for Mermaid).
- **Config examples:** When added, use `*.example.yaml` or `*.example.json`; never commit secrets.

---

**See also:** [architecture.md](architecture.md), [implementation/README.md](../implementation/README.md).
