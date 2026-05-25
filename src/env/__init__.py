"""OpenEnv-spec environment for peptide / protein design.

To be implemented in Phase 2. Mirrors PharmaRL's env design:
  - FastAPI server, session-keyed, Pydantic-typed
  - Action schema: ADD_RESIDUE, SUBSTITUTE_RESIDUE, MUTATE_REGION, TERMINATE
  - Observation: current sequence, target receptor, valid-action set, per-component reward
"""
