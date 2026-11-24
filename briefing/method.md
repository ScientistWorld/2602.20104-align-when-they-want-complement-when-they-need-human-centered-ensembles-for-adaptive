# Paper's Method

Paper: "Align When They Want, Complement When They Need! Human-Centered
Ensembles for Adaptive Human-AI Collaboration" (Amin, Yin, Khanna, AAAI 2026).

## Team model

Confidence-Gated Probabilistic Reliance (CGPR, Eq. 2):
- a human whose confidence C^h(x) exceeds the threshold tau keeps their own
  judgement;
- otherwise they follow the AI with probability r = 1 − L_h(D_a, m), the
  AI's agreement with the human on the alignment region D_a = {C^h > tau}.

With tau drawn per decision from a distribution F_T, region membership is
probabilistic (Proposition 5). The paper's instance weights are
w_a = F_T(C^h) and w_c = 1 − F_T(C^h).

## Adaptive ensemble

- **Aligned specialist m_a.** Minimises disagreement with the human's own
  predictions, with the alignment-region weights.
- **Complementary specialist m_c.** Minimises prediction loss on the ground
  truth, with the complementarity-region weights.
- **Routing.**
  - Oracle routing (Eq. 7) sends a case to m_a when it is in the alignment
    region and to m_c otherwise.
  - The Rational Routing Shortcut (RRS, Eq. 8) sends a case to the more
    confident specialist instead.

## Implementation in this gym (`method/adaptive_team_ai.py`)

This follows the paper's synthetic recipe: logistic regression for every AI
model, sklearn defaults, deterministic.

- **Group given (`admission_group_known*`).** This is the paper's synthetic
  default, where group membership is treated as known.
  - m_a is fitted on the privileged group's human judgements, with weights
    F_T(C^h).
  - m_c is fitted on the other group's outcomes, with weights 1 − F_T(C^h).
  - Routing uses the group.
- **Group hidden (`admission_group_hidden*`).** The paper's
  behaviour-grounded protocol.
  - m_a and m_c are fitted on all training rows with Proposition 5 weights.
  - The human's confidence is unobservable at test time. As in the paper's
    pipeline, a confidence model fitted on the training (pilot) data
    estimates P(C^h > tau | x), and Eq. 7 routes to m_a when that estimate
    is at least 0.5.
  - `--router rrs` uses the paper's RRS instead. On this world it scores
    clearly lower; see HARDENING.md.
- **Baselines.** The paper's single-model paradigms, also logistic
  regression:
  - standard AI (ERM on outcomes);
  - behaviour-aware single AI (the weighted surrogate g_p of Theorem 4, with
    human targets in the alignment region);
  - aligned AI alone;
  - complementary AI alone.

`method/college_admission_method.py` and `method/aaai26-adaptive-ai/` (the
authors' release) reproduce the paper's own two-feature College Admission
sweep. They feed the unscored `scoring/paper_reproduction.json`, which matches
the authors' released numbers exactly.
