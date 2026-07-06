[RULE 001] PDE blow-up integral power selection
content: When determining the strongest provable blow-up criterion for nonlinear PDEs, do not default to the L1-in-time gradient norm; verify if the specific energy estimates or equation structure (e.g., 2D non-local systems) establish the divergence of the L2-in-time gradient norm as the necessary condition.
why: The agent incorrectly selected the standard L1 gradient integral (Choice C) for a 2D non-local PDE where the proven necessary condition is the L2 gradient integral (Choice D).

[RULE 002] Selection of stronger result options
content: When a multiple-choice question asks for the "strongest statement" or "best provable result" and offers a meta-option claiming a stronger result exists, select the meta-option, as it typically indicates the existence of an asymptotic formula or tighter bound not listed among the specific distractors.
why: The agent failed to select the meta-option in a square-full number counting task because it got distracted by verifying specific exponents; specifying the "strongest statement" trigger helps the agent recognize this pattern and avoid getting trapped by plausible but non-optimal specific bounds.

[RULE 003] Harmonic Analysis critical variation unboundedness
content: When evaluating the L^p boundedness of r-variation norms for circular averaging operators in R^2 with r=2, do not assume boundedness for p > 2 based on maximal function results; recognize that the 2-variation operator is unbounded for all p >= 1.
why: The agent incorrectly selected options implying boundedness for p > 2 (Choices A, E) or partial boundedness, failing to recognize that the critical geometry of R^2 combined with the critical variation index r=2 leads to global unboundedness for all p.

[RULE 004] Block-stable graph tree criticality
content: When analyzing the block tree distribution of random graphs from a block-stable class relative to the critical threshold u_C, recognize that the tree is subcritical for u < u_C but remains critical for all u >= u_C, rather than becoming supercritical.
why: The agent incorrectly assumed the block tree becomes supercritical for u > u_C (Teacher, Option A) or always critical due to conditioning (Student, Option B), failing to recognize that for block-stable classes, the critical regime extends to all u >= u_C (Option D).
