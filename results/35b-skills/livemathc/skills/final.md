[RULE 001] Stronger-result meta-options
content: When a multiple-choice problem includes a meta-option claiming another option is correct but a stronger result holds, do verify that ordinary options do not add unsupported extra claims, then choose the meta-option if a remaining true option is only a weaker consequence of the strongest available conclusion.
why: The failing rollout chose an ordinary option that added a false extra strengthening, while the succeeding rollout rejected that overstrong option and chose the meta-option because another ordinary option was a true weaker consequence of the stronger theorem.

[RULE 002] Squared-gradient blow-up criterion
content: When a PDE multiple-choice problem about a smooth nonlocal two-dimensional evolution equation in an anisotropic Sobolev well-posedness setting asks for the strongest necessary blow-up behavior at a finite maximal existence time, do choose the statement that the time integral of the squared full spatial-gradient supremum norm diverges rather than defaulting to a first-power gradient criterion.
why: The failing rollout defaulted to the familiar first-power Beale-Kato-Majda criterion and the teacher chose a subquadratic option, while the correct theorem for this setting gives divergence of the squared full-gradient supremum norm.

[RULE 003] Finite-cover character induction
content: When a multiple-choice problem asks about the natural direct-image or induction map from a finite cover on complex general-linear character varieties with Goldman-type Poisson structures, do choose the Poisson embedding statement without strengthening it to an isomorphism onto a symplectic leaf or weakening it because of presumed noninjectivity.
why: The failing rollouts either overstrengthened the theorem to an isomorphism onto a symplectic leaf or weakened it to a noninjective Poisson map, while the correct result is that the direct-image induction map is a Poisson embedding.

[RULE 004] Critical circular variation unboundedness
content: When a harmonic-analysis multiple-choice problem asks for the strongest uniform Lp statement for the critical 2-variation of planar circular averaging over a fixed interval on Schwartz inputs, do choose total unboundedness for every p≥1, treating frequency-localized logarithmic estimates as weaker non-uniform substitutes.
why: The failing rollout imported p>2 spherical maximal boundedness and chose a frequency-localized logarithmic estimate, while the succeeding rollout selected the theorem that the critical 2-variation is not uniformly Lp bounded for any p.

[RULE 005] (1,1) surgery L-space relations
content: When a low-dimensional topology multiple-choice problem asks which L-space conjecture implications are known for closed manifolds obtained by Dehn surgery on genus-one one-bridge knots, do choose that non-L-space is equivalent to admitting a coorientable taut foliation and that left-orderability implies non-L-space, without assuming full three-way equivalence.
why: The failing rollout assumed the full L-space conjecture was proven for this surgery class and chose three-way equivalence, while the succeeding rollout selected the partial known relation that non-L-space is equivalent to taut foliation and left-orderability implies non-L-space.

[RULE 006] Singular anisotropic elliptic exponents
content: When a multiple-choice problem asks for the strongest existence and anisotropic Sobolev regularity theorem for positive distributional solutions of a singular anisotropic elliptic problem with L1 data, a power-dependent principal coefficient, and a subunit singular gradient absorption term, do choose the piecewise statement with the first regime strictly below the threshold, the intermediate regime stated using auxiliary admissible exponents rather than the endpoint formula, and the superlinear regime in the natural anisotropic space.
why: The failed retry used a similar rule but still chose the option with the closed threshold and explicit endpoint-style intermediate exponents, while the correct choice kept the first threshold open and used auxiliary admissible exponents in the intermediate regime.

[RULE 007] VMO adjoint elliptic regularity
content: When a multiple-choice problem asks for the strongest regularity of a locally bounded distributional solution to a linear elliptic adjoint or non-divergence equation with locally VMO uniformly elliptic coefficients, coefficient divergence in L2, and L1/L2 forcing, do choose the qualitative local H^{1,2} membership statement rather than an option asserting a quantitative estimate depending only on the forcing.
why: The failing rollout selected the stronger-looking H^{1,2} option with an unsupported a priori estimate depending only on the forcing, while the succeeding rollout chose the theorem's guaranteed qualitative H^{1,2}_{loc} regularity alone.

[RULE 008] WUA ultrafilter order separation
content: When a set-theory multiple-choice problem asks about consistency of Lipschitz-below but Ketonen-incomparable countably complete ultrafilters under weak ultrapower assumptions, do choose the option saying the separation is consistent even together with the weak ultrapower axiom rather than assuming that axiom forces Ketonen comparability.
why: The failing rollout chose only the bare relative consistency result after assuming WUA makes Ketonen-incomparability impossible, while the succeeding rollout selected the stronger statement that the same separation can consistently occur with WUA.

[RULE 009] Block-stable block-tree criticality
content: When a multiple-choice problem asks about threshold behavior for block-weighted rooted connected structures from a block-stable graph class and the associated tree law, do choose that the generating-function singularity changes at the threshold and that the block tree is subcritical below the threshold but critical at and above it, rather than critical for all weights or supercritical above the threshold.
why: The failing rollout chose the option saying conditioning on size makes the block tree critical for every weight, while the succeeding rollout chose the theorem that it is subcritical below the threshold and critical at and above the threshold.

[RULE 010] Square-full triple estimates
content: When an analytic-number-theory multiple-choice problem asks for the strongest quantitative estimate for counting additive triples of square-full positive integers with a bounded total term and includes both a meta-option and a true half-density upper bound with an epsilon loss, do choose the meta-option because a sharper result than that listed upper bound is provable.
why: The failing rollouts first chose a weaker power-saving bound and then the true epsilon-loss half-density bound, while the successful answer recognized the listed bound was only a consequence of a still stronger theorem and therefore selected the meta-option.
