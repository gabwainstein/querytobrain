# NeuroLab Non-Goals

These are explicitly **not** goals for NeuroLab. If a request pushes in this direction, it is out of scope unless product docs are updated.

## MVP non-goals

1. **Full raw preprocessing in chat time**  
   We do not run fMRIPrep (or equivalent) on arbitrary OpenNeuro/BIDS data during a user session. Meta-analytic and pre-computed datasets only for MVP.

2. **Clinical or medical advice**  
   No clinical decision support, diagnosis, treatment recommendation, or individualized medical advice. Educational and research context only.

3. **Causal claims from overlap alone**  
   Spatial overlap of maps is not sufficient to claim causation. We support mechanistic *hypotheses* and “how to validate,” not causal conclusions without intervention/longitudinal evidence.

4. **Controlled substances**  
   We do not provide or recommend controlled substances.

5. **Reimplementing a full orchestration layer**  
   NeuroLab is not intended to replace a full external orchestrator, workflow engine, or analysis platform. It focuses on neuroscience tools, maps, and evidence-aware interfaces.

6. **Real-time raw pipeline orchestration**  
   No on-the-fly “run fMRIPrep on this OpenNeuro dataset” in MVP. That is Phase 2.

7. **Medical/clinical endorsement via token or product**  
   Any governance or token mechanics must not be framed as medical or clinical endorsement.

---

Historical source material is archived under `docs/external-specs/` and is not part of the default publish surface for this personal repo.
