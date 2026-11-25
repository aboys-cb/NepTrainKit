# Defect and magnetic workflows

## Defects

- Ask which defect, what sites or regions, how many per structure or at what concentration, and whether a pristine reference is needed. For dilute vacancies, suggest one vacancy per supercell. Size the cell for concentration and periodic-image separation. Check site availability, minimum distances for insertions, and upper bounds on unique random outputs.
- Expand before placing defects. Use `VacancyDefectCard` for unrestricted vacancies and `RandomVacancyCard` for element or group targets. Use `RandomDopingCard` for substitutions and `InsertDefectCard` for random interstitials or surface adsorbates. For stacking-fault scans, use `StrictGSFEPathCard` on a cell oriented for the slip plane and direction.
- Generate independent defect types from the same parent unless the user wants them in the same structure. To sample one defect configuration under multiple strains, place the defect once, then deform and optionally perturb atoms. If placement is defined by distances in the final strained cell, deform first and apply the placement constraints there. Keep random perturbations off controlled defect paths unless requested.

## Magnetism

- Ask whether the goal is comparing magnetic orders, sampling around a reference state, or measuring a controlled response. Confirm the reference moments, target elements or sites, and collinear or vector format.
- For new AFM patterns or finite-q textures, expand before assigning the new spins. Repeating an already magnetic cell tiles its spin pattern. Choose repeats that fit the required sublattices or wavelengths and verify periodic closure. A fixed reciprocal-cell index changes physical q when cell size changes. For size comparisons at fixed Cartesian q, use commensurate cells and generate the texture on each.
- Use `MagneticOrderCard` to generate FM, AFM, or disordered initial states independently on the same geometry. For sampling around a reference, reuse valid moments or initialize them with `SetMagneticMomentsCard`. Use `SpinDisorderCard` for partial flips or reorientation and `MagneticMomentRotationCard` for random angular or magnitude perturbations.
- Use `LocalMagneticResponseCard` for site, pair, or group response, `MagnetoelasticResponseCard` for paired strain–spin scans, and `SOCTextureResponseCard` for anisotropy or finite-q textures. Preserve whole `response_group` sets during selection and export. Generated moments are DFT initial or constraint targets. Use converged DFT moments for training labels.
- For combined sampling, establish geometry and composition first, then initialize or check moments and apply spin variants. If a magnetic pattern must carry through, preserve it and verify site mapping and moments after deletion, insertion, or substitution. For pristine–defect comparisons, branch from the same supercell and apply the same magnetic protocol to both paths, keeping their outputs separate.
