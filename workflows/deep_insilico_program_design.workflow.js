export const meta = {
  name: 'deep-insilico-program-design',
  description: 'Scope the deepest reality-calibrated in-silico program for recognition-gated DR5 apoptosis (EARM kinetics, agonism-by-clustering predictor, PhysiCell tissue, patient-data grounding) — tools, real parameters, calibration datasets',
  phases: [
    { title: 'Research', detail: 'parallel: EARM kinetics, DR5 agonism/clustering + calibration set, PhysiCell tissue, patient/priming datasets' },
    { title: 'Synthesize', detail: 'concrete Colab-feasible multi-scale build plan, each rung calibrated on named real data' },
    { title: 'Critique', detail: 'adversarial: is each rung honestly calibratable, or self-deception/overclaim?' },
  ],
}

const CTX = `PROJECT: cancer-recon-apoptosis (Anshuman Atrey). Hypothesis: a RECOGNITION-GATED, SELF-
PROPAGATING apoptosis wave makes cancer destroy itself from within (Shriya Rai's concept), gated by a
cancer antigen (anchor = Trop2/TACSTD2) and triggered by clustering the death receptor DR5 (TNFRSF10B).
We have abstract agent-based sims (recognition-gating is the safe+lethal regime; grow-then-reverse;
metastasis scout+amplify). VERIFIED CONSTRAINT (do NOT contradict): wet-lab cannot be skipped; no
purely-computational drug exists; AI de-risks chemistry not biology; the AGONISM step (does a DR5
binder CLUSTER it to fire caspase-8) is the crux with no naive in-silico predictor.

GOAL NOW: design the MAXIMUM reality-calibrated IN-SILICO program runnable on a LAPTOP / GOOGLE COLAB
(free tier, CPU; some GPU ok). No wet-lab. We want to prove as much as silicon honestly allows, and to
CALIBRATE every rung against REAL public data so results are falsifiable (the way we calibrated a
safety audit against known drug targets). Any dataset/model is fair game. Be concrete: exact tools,
install on Colab, REAL parameter values + their sources, and named calibration datasets. Distinguish
what each rung can PROVE vs what stays a proxy needing a bench. No hype.`

const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    topic: { type: 'string' },
    bottom_line: { type: 'string' },
    tools: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { name: {type:'string'}, what: {type:'string'}, colab_install: {type:'string'},
        api_or_usage: {type:'string'}, caveats: {type:'string'} },
      required: ['name','what','colab_install','api_or_usage','caveats'] } },
    real_parameters_or_datasets: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { name: {type:'string'}, value_or_access: {type:'string'}, source: {type:'string'} },
      required: ['name','value_or_access','source'] } },
    calibration_set: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { item: {type:'string'}, known_label: {type:'string'}, source: {type:'string'} },
      required: ['item','known_label','source'] } },
    what_it_can_prove: { type: 'string' },
    what_stays_proxy: { type: 'string' },
    citations: { type: 'array', items: { type: 'string' } },
  },
  required: ['topic','bottom_line','tools','real_parameters_or_datasets','calibration_set','what_it_can_prove','what_stays_proxy','citations'],
}

phase('Research')
const [earm, agonism, physicell, patient] = await parallel([
  () => agent(CTX + `

YOUR TASK (MOLECULAR KINETICS — PySB/EARM): Detail the Extrinsic Apoptosis Reaction Model (EARM;
Albeck/Spencer/Sorger 2008; Lopez et al. PySB) and PySB. Exact Colab install (pip pysb + BioNetGen/
the bundled simulators); how to load EARM (is it bundled in pysb.examples / earm package?); the real
rate constants and initial protein counts the model ships with; the key readouts (caspase-3 substrate
'PARP' cleavage, time-to-death, the documented ALL-OR-NONE / bistable MOMP switch, snap-action). MOST
IMPORTANT: how do we set the UPSTREAM INPUT that represents our recognition-gated DR5/TRAIL activation
(ligand dose / activated caspase-8) so we can sweep 'DR5 activation level' -> does the cell commit to
death, and what threshold/timing? Give a minimal runnable code sketch. Calibration: EARM is itself
fit to real single-cell time-lapse death-timing data (cite). What can this PROVE (death-commitment
dynamics under real kinetics) vs stay proxy?`,
    { label: 'earm-kinetics', phase: 'Research', schema: RESEARCH_SCHEMA, agentType: 'general-purpose' }),

  () => agent(CTX + `

YOUR TASK (DR5 AGONISM = CLUSTERING GEOMETRY + a CALIBRATION SET — the crux). (1) Summarise the
structural mechanism of DR5 agonism: higher-order clustering / lattice formation (cryo-EM and the
Vanamee & Faustman hexagonal-lattice / 'cluster of clusters' model of TNFRSF receptors); how VALENCY
and GEOMETRY (monovalent vs bivalent vs tri/hexavalent, and FcgammaR crosslinking dependence)
determine agonist potency. (2) Build a CALIBRATION SET of DR5 agonists/binders with KNOWN agonist
potency we can score a geometry/valency-based predictor against: e.g. TRAIL (trimeric, agonist),
hexavalent IgM agonist (IGM-8444 / aplitabart — potent, crosslinking-independent), bivalent agonist
mAbs (conatumumab/AMG655, drozitumab, tigatuzumab — agonism FcgammaR-crosslinking-dependent),
monovalent Fab / non-crosslinked (weak/inert). For each: valency, crosslinking dependence, relative
agonist potency, citation. (3) What computational features (valency, epitope, inter-receptor spacing
from a modelled complex, ability to tile the lattice) plausibly PREDICT agonism, and what tools
(AF3/Boltz multi-DR5 complexes, coarse-grained clustering models, simple valency/geometry scores) can
compute them on Colab. Be honest: this is a calibratable PROXY, not proof of agonism.`,
    { label: 'agonism-clustering', phase: 'Research', schema: RESEARCH_SCHEMA, agentType: 'general-purpose' }),

  () => agent(CTX + `

YOUR TASK (TISSUE DYNAMICS — PhysiCell, realistic). PhysiCell (Macklin lab) agent-based 3D multicellular
simulator: Colab feasibility (it is C++/OpenMP — can it build+run in Colab? PhysiCell Studio? the
pcdl / Python loaders? or a lighter Python ABM like a BioFVM/diffusion + cells grid?). REAL parameters
to ground a tumour + death-signal-propagation model: cancer cell diameter (~15-20 um), cell-cycle/
doubling time, apoptosis death rate + duration (PhysiCell defaults / literature), substrate (death
ligand / DAMP) DIFFUSION COEFFICIENT and decay, oxygen for viable-rim. How to encode: a diffusing
death signal + recognition-gated death (gated by an antigen field) + measure propagation velocity,
clearance, and CONTAINMENT (does it leak to healthy). Real tumour geometry sources (e.g. from imaging
or seeded spheroid). Cite parameter sources. What can this PROVE (mechanism coherence + parameter-
sensitivity at tissue scale) vs proxy?`,
    { label: 'physicell-tissue', phase: 'Research', schema: RESEARCH_SCHEMA, agentType: 'general-purpose' }),

  () => agent(CTX + `

YOUR TASK (PATIENT/DATA GROUNDING — apoptotic priming + where it works). Identify REAL public datasets
to ground WHICH cancers/patients are most susceptible to a DR5/extrinsic-apoptosis trigger, so the
model predicts 'works best here' rather than a generic claim. Cover: (1) DR5 (TNFRSF10B), caspase-8,
cFLIP/CFLAR, BCL2-family, decoy receptors (TNFRSF10C/D) expression across cancers — TCGA / GDC, DepMap/
CCLE, CELLxGENE (we already pull scRNA). (2) The concept of APOPTOTIC PRIMING / BH3 profiling (Letai
lab) — what it measures (how close a cell is to the apoptotic threshold) and any COMPUTATIONAL analog
or dataset (e.g. gene-expression signatures of priming, DepMap dependency on BCL2 family). (3) DepMap
to check if cancer lines depend on anti-apoptotic genes (a priming proxy). For each dataset: exact
access (API/download) usable on Colab + what it lets us compute. Cite. What can this PROVE (patient-
stratification hypothesis grounded in real expression) vs proxy?`,
    { label: 'patient-priming', phase: 'Research', schema: RESEARCH_SCHEMA, agentType: 'general-purpose' }),
])

phase('Synthesize')
const PLAN_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    headline: { type: 'string' },
    build_rungs: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: {
        rung: {type:'string'}, builds_what: {type:'string'}, tools: {type:'string'},
        real_calibration: {type:'string'}, colab_feasibility: {type:'string'},
        proves: {type:'string'}, stays_proxy: {type:'string'}, order: {type:'integer'} },
      required: ['rung','builds_what','tools','real_calibration','colab_feasibility','proves','stays_proxy','order'] } },
    agonism_predictor_verdict: { type: 'string' },   // is the clustering-geometry agonism predictor legit + calibratable?
    biggest_risks: { type: 'array', items: { type: 'string' } },
    what_the_whole_program_can_honestly_claim: { type: 'string' },
    recommended_first_build: { type: 'string' },
  },
  required: ['headline','build_rungs','agonism_predictor_verdict','biggest_risks','what_the_whole_program_can_honestly_claim','recommended_first_build'],
}
const plan = await agent(CTX + `

Synthesise the four research outputs into ONE concrete, ordered, Colab-feasible multi-scale BUILD PLAN
for the deep in-silico program. Each rung: what it builds, tools (+ Colab install), the REAL data it is
calibrated against, what it can PROVE vs what stays proxy. Give a clear verdict on whether the
clustering-geometry AGONISM PREDICTOR is a legitimate, calibratable proxy (the crux). State honestly
what the whole program can claim. Recommend the first build (the user wants realistic kinetics first).

EARM: ${JSON.stringify(earm)}
AGONISM: ${JSON.stringify(agonism)}
PHYSICELL: ${JSON.stringify(physicell)}
PATIENT: ${JSON.stringify(patient)}`,
  { label: 'synthesis', phase: 'Synthesize', schema: PLAN_SCHEMA })

phase('Critique')
const CRITIQUE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    is_honest_and_buildable: { type: 'boolean' },
    rungs_that_overclaim: { type: 'array', items: { type: 'string' } },
    agonism_predictor_real_or_handwaving: { type: 'string' },
    colab_feasibility_problems: { type: 'array', items: { type: 'string' } },
    required_fixes: { type: 'array', items: { type: 'string' } },
    verdict: { type: 'string' },
  },
  required: ['is_honest_and_buildable','rungs_that_overclaim','agonism_predictor_real_or_handwaving','colab_feasibility_problems','required_fixes','verdict'],
}
const critique = await agent(CTX + `

Adversarially stress-test the build plan. We are pushing in-silico to its max WITHOUT faking. Attack:
(1) Does any rung overclaim what it proves (esp. calling proxy results 'evidence the therapy works')?
(2) Is the clustering-geometry AGONISM predictor genuinely calibratable against the known agonist set,
or is it hand-waving dressed up — would it actually separate known agonists from inert binders, and is
the calibration set real and labelled? (3) Colab free-tier feasibility problems (PhysiCell C++ build,
RAM, GPU, big downloads)? (4) Is each 'real calibration' actually real and sufficient, or cosmetic?
List required fixes. Default to skepticism; pass only if the plan is honest AND buildable on Colab.

PLAN: ${JSON.stringify(plan)}
AGONISM RESEARCH: ${JSON.stringify(agonism)}`,
  { label: 'adversarial-critique', phase: 'Critique', schema: CRITIQUE_SCHEMA })

return { plan, critique }
