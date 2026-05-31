# Paper's Method

The paper introduces Confidence-Gated Probabilistic Reliance (CGPR), a model of
AI-assisted decision making where high-confidence humans keep their own answer
and low-confidence humans accept AI advice with a probability determined by the
AI's agreement with humans in the high-confidence region.

The proposed system trains two specialists. The aligned specialist minimizes
disagreement with human predictions in the alignment region. The complementary
specialist minimizes prediction loss against ground truth in the complementary
region. At inference, an adaptive ensemble selects between specialists either
with oracle region membership or with the Rational Routing Shortcut (RRS), which
chooses the specialist with higher model confidence.

The implementation in `method/college_admission_method.py` follows these steps
with logistic-regression specialists, a behavior-aware single model, CGPR team
decisions, oracle routing, and RRS routing.
