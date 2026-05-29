export const meta = {
  name: 'insilico-feasibility-and-track-record',
  description: 'Honestly assess: do realistic virtual-cell models exist that could replace wet-lab, and the real track record of in-silico -> physically-real drug discovery',
  phases: [
    { title: 'Research', detail: 'parallel: virtual-cell models, drug track-record, reliability map, wet-lab necessity' },
    { title: 'Synthesize', detail: 'honest answer + implications for our project' },
    { title: 'Critique', detail: 'adversarial: de-hype, verify real clinical status of every claimed success' },
  ],
}

const CTX = `CONTEXT: cancer-recon-apoptosis — a COMPUTATIONAL cancer-therapy research project (Anshuman Atrey).
We design a recognition-gated self-propagating apoptosis mechanism (operationalising Shriya Rai's
"cancer destroys itself from within"). We have abstract simulations; we are deciding how far in-silico
work can take us toward a REAL discovery.

THE USER'S QUESTION (answer honestly, no hype): Are there computational models of cells/molecules
realistic enough that we can test interventions IN SILICO and largely AVOID wet-lab experiments —
i.e. "every action/scenario a cell can have, including how it reacts to a drug, is already captured in
the model"? He cites AlphaFold as an example of this intuition. And: how many in-silico/AI discoveries
actually became REAL, physically-available drugs in the lab/clinic — and did any skip wet-lab?

MANDATE: real status only (trial phase / approved / discontinued), cite sources, distinguish validated
fact from press-release/marketing. We must NOT overclaim. Knowledge cutoff ~2026; use web search.`

const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    topic: { type: 'string' },
    bottom_line: { type: 'string' },
    key_points: { type: 'array', items: { type: 'string' } },
    examples: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: {
        name: { type: 'string' },
        what_was_computational: { type: 'string' },
        real_world_status: { type: 'string' },        // e.g. "Phase II", "approved", "discontinued Phase I", "animal only"
        skipped_wetlab: { type: 'boolean' },
        notes: { type: 'string' },
        citation: { type: 'string' },
      },
      required: ['name','what_was_computational','real_world_status','skipped_wetlab','notes','citation'] } },
    limitations_or_caveats: { type: 'array', items: { type: 'string' } },
  },
  required: ['topic','bottom_line','key_points','examples','limitations_or_caveats'],
}

phase('Research')
const [vcell, track, reliab, wetlab] = await parallel([
  () => agent(CTX + `

YOUR TASK (VIRTUAL-CELL / WHOLE-CELL MODELS): What is the actual state of the art in computational
models that predict how a CELL responds to perturbations/drugs? Cover: (1) Karr et al. 2012
Mycoplasma genitalium whole-cell model (first complete-organism sim) — scope + why human cells are
far harder; (2) the "AI Virtual Cell" initiative (Bhatt/Quake et al., "How to build a virtual cell",
Cell 2024; CZI; Arc Institute) — what it aspires to vs what exists; (3) foundation models scGPT,
Geneformer, scFoundation, Arc 'State' — can they predict response to UNSEEN drugs/perturbations, and
how accurately?; (4) perturbation predictors (GEARS); (5) mechanistic models (PhysiCell, PySB/EARM,
Recon3D metabolic). The decisive question: is there ANY model today where a human cell's reaction to
a NOVEL drug is reliably 'already captured' such that wet-lab can be skipped — or is that aspirational?
Be brutally honest about current predictive accuracy and generalisation to novel interventions.`,
    { label: 'virtual-cell-models', phase: 'Research', schema: RESEARCH_SCHEMA, agentType: 'general-purpose' }),

  () => agent(CTX + `

YOUR TASK (TRACK RECORD — in-silico/AI to REAL drug): List the most cited examples of computationally
/ AI-discovered or -designed drugs and their REAL current status. For EACH give real_world_status
(preclinical / Phase I/II/III / approved / discontinued) and skipped_wetlab (almost certainly false):
Insilico Medicine rentosertib/INS018_055 (IPF); Exscientia DSP-1181 (OCD, first AI-designed in human
trials) and others; BenevolentAI baricitinib for COVID (repurposing — was it actually authorised/used?);
halicin & abaucin (MIT/Collins ML-discovered antibiotics — validated where?); Baker-lab de novo designed
proteins/binders reaching clinic; Recursion/Relay/Schrodinger clinical-stage assets; any AlphaFold-
enabled drug milestone. CRITICAL: is there ANY AI-designed-from-scratch drug that is FDA/EMA APPROVED
yet (as of 2026)? Distinguish 'in trials' from 'approved'. Cite each.`,
    { label: 'drug-track-record', phase: 'Research', schema: RESEARCH_SCHEMA, agentType: 'general-purpose' }),

  () => agent(CTX + `

YOUR TASK (RELIABILITY MAP — what can in-silico be TRUSTED for): Map where computational prediction is
reliable enough to ACT ON without wet-lab confirmation vs where it is NOT. Cover: protein structure
(AlphaFold2/3 — high reliability, used in real pipelines, but caveats: disordered regions, novel folds,
conformational change); protein-protein/ligand binding & affinity (proxy, not measured Kd — accuracy?);
ADMET prediction; cell-level drug EFFICACY; in-vivo response; toxicity; receptor AGONISM vs binding (a
known failure mode). For each, state the honest reliability tier and whether a 'yes' from the model can
be trusted without experiment. This tells us which of OUR predictions (binding via Boltz, apoptosis sim)
are trustworthy and which are merely hypothesis-generating.`,
    { label: 'reliability-map', phase: 'Research', schema: RESEARCH_SCHEMA, agentType: 'general-purpose' }),

  () => agent(CTX + `

YOUR TASK (WET-LAB NECESSITY + realistic role): Answer directly: has ANY therapeutic reached patients
via PURE computation with NO wet-lab / animal / clinical validation? (Search hard; the answer is likely
NO — establish it with evidence.) Then quantify the REALISTIC role of in-silico: how much does it
compress timelines/cost / improve hit rates (cite any numbers, e.g. Insilico's target-to-preclinical
timeline, AI-drug Phase I success-rate analyses ~2024-2025)? And the sobering side: do AI-discovered
drugs still face normal ~90% clinical attrition; any high-profile AI-drug FAILURES (e.g. Exscientia/
Sumitomo DSP-1181 discontinuation, BenevolentAI atopic-dermatitis fail)? Conclude with the realistic
contribution ceiling for a computational-only academic project (de-risk + prioritise + a defined
wet-lab handoff), honestly stated.`,
    { label: 'wetlab-necessity', phase: 'Research', schema: RESEARCH_SCHEMA, agentType: 'general-purpose' }),
])

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    direct_answer_can_we_skip_wetlab: { type: 'string' },
    virtual_cell_reality: { type: 'string' },
    track_record_summary: { type: 'string' },
    any_approved_pure_insilico_drug: { type: 'string' },   // honest yes/no + evidence
    what_we_can_trust_in_our_project: { type: 'array', items: { type: 'string' } },
    what_we_cannot: { type: 'array', items: { type: 'string' } },
    realistic_ceiling_for_our_project: { type: 'string' },
    recommended_path: { type: 'array', items: { type: 'string' } },
    headline_numbers: { type: 'array', items: { type: 'string' } },
  },
  required: ['direct_answer_can_we_skip_wetlab','virtual_cell_reality','track_record_summary',
             'any_approved_pure_insilico_drug','what_we_can_trust_in_our_project','what_we_cannot',
             'realistic_ceiling_for_our_project','recommended_path','headline_numbers'],
}
const synthesis = await agent(CTX + `

Synthesise the four research outputs into an HONEST, cited answer to the user's question, plus concrete
implications for cancer-recon-apoptosis. Be direct about whether wet-lab can be skipped (it cannot, but
say why and what in-silico DOES buy us), the virtual-cell reality (aspirational vs existing), the real
track record with the key fact (is any AI-designed-from-scratch drug APPROVED yet?), what WE can/can't
trust, our realistic contribution ceiling, and the recommended path. Include headline numbers/examples.

VIRTUAL-CELL: ${JSON.stringify(vcell)}
TRACK-RECORD: ${JSON.stringify(track)}
RELIABILITY: ${JSON.stringify(reliab)}
WET-LAB NECESSITY: ${JSON.stringify(wetlab)}`,
  { label: 'synthesis', phase: 'Synthesize', schema: SYNTH_SCHEMA })

phase('Critique')
const CRITIQUE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    is_honest_not_hyped: { type: 'boolean' },
    overclaims_found: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { claim: { type: 'string' }, problem: { type: 'string' }, correction: { type: 'string' } },
      required: ['claim','problem','correction'] } },
    status_corrections: { type: 'array', items: { type: 'string' } },  // any drug-status claims that are wrong/stale
    verdict: { type: 'string' },
  },
  required: ['is_honest_not_hyped','overclaims_found','status_corrections','verdict'],
}
const critique = await agent(CTX + `

Adversarially fact-check the synthesis below. This space is FULL of hype and press releases. Attack it:
(1) Are any cited 'successes' actually discontinued, stalled, or marketing rather than real clinical
milestones (e.g. DSP-1181 was discontinued; check baricitinib's actual COVID authorisation status;
check whether any 'AI drug' is truly APPROVED vs just in trials)? (2) Does it overstate virtual-cell
capability or understate wet-lab necessity? (3) Is the 'realistic ceiling' for our project honest, or
does it secretly imply we can do more than we can? Correct every overclaim and wrong status. Default to
skepticism; only pass it if it is genuinely de-hyped and accurate.

SYNTHESIS: ${JSON.stringify(synthesis)}
RAW TRACK-RECORD: ${JSON.stringify(track)}`,
  { label: 'adversarial-critique', phase: 'Critique', schema: CRITIQUE_SCHEMA })

return { synthesis, critique }
