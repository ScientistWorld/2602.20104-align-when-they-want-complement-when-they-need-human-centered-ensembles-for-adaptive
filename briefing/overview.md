# Overview

Paper: "Align When They Want, Complement When They Need! Human-Centered
Ensembles for Adaptive Human-AI Collaboration" by Hasan Amin, Ming Yin, and
Rajiv Khanna.

This reproduction packages the core synthetic experiment: a two-region admission
task where human confidence and ability differ by region. The method trains
aligned and complementary specialists and adaptively routes between them under
the CGPR human-AI team model. The benchmark scores whether adaptive
recommendations improve team accuracy while preserving the reliance quantity
that makes humans willing to use AI advice.

The packaged reproduction reaches `secondary_claims`: the alpha=0.9 core
setting matches the authors' reported synthetic result closely, and the
human-reliability sweep at alpha values 0.5, 0.7, 0.8, 0.9, and 1.0 preserves
the same adaptive-routing advantage.
