Overall Algorithm
=================

The overall estimation algorithm performs steps in the following order:
1. Unadjusted Design Weight
2. Design Weight
3. Calibration Weight

* All calculations are done on a per-period basis so that any references to
Strata and Calibration Group refer to those values within a given period in
a multi-period dataset.

Estimation Types
----------------

* Expansion Estimation - calculates the Design Weight by Strata using Unadjusted
  Design Weight and, optionally, Birth/Death adjustment.
* Ratio Estimation - calculates the Calibration Weight. There are two types:
  * Separate Ratio Estimation - calculates Calibration Weight by Strata.
  * Combined Ratio Estimation - calculates Calibration Weight by Calibration
    Group.

Required Inputs
---------------

Depending on the desired type of Estimation there are different required inputs.
For Expansion Estimation without Birth/Death adjustment, only Period, Strata
and Sample Marker are required. For Birth/Death adjustment Death Marker and
H Value is required.

For Separate Ratio Estimation, an Auxiliary Value is required. In order to
perform Combined Ratio Estimation a Calibration Group is also required.

Unadjusted Design Weight
========================

Estimation calculates this first as it is used in all types of estimation.
It uses the Sample Marker which must only contain a `0` or a `1`. This
allows it to work out the total population as the count of the Sample Marker
and the sample population as the sum of the Sample Marker.

The calculation is:
```
population_count = count(sample_marker)
sample_count = sum(sample_marker)
unadjusted_design_weight = population_count/sample_count
```

Design Weight
=============

Taking the calculated `unadjusted_design_weight`, Estimation then uses
birth/death adjustment to get the final Design Weight. The calculation uses
the sample population, number of deaths in the population and the
birth/death adjustment parameter (`h_value`).

Using the definitions above, the calculation is:
```
death_count = sum(death_marker)
death_factor = 1 + (h_value * (death_count/(sample_count - death_count)))
design_weight = unadjusted_design_weight * death_factor
```

The h value is usually a 1 or 0 but could possibly be any positive number.
For example, if a stratum is dying then the value may be between 0 and 1,
however if a stratum is growing the value may be greater than 1.

In the event that no h value is specified then 0 will be used. This means
that the Unadjusted Design Weight will be adjusted by a factor of 1 and thus
no adjustment will be performed.

Calibration Weight
==================

Estimation uses the `unadjusted_design_weight` again and uses the Auxiliary
value for the responders to turn it into a Calibration Weight. Note that in
the calculation below, the numerator is calculated from the whole population
of a grouping (strata or calibration group), whereas the denominator is
calculated for the sampled subset of that grouping.

```
aux_design = unadjusted_design_weight * auxiliary_value * sample_marker
calibration_weight = sum(auxiliary_value)/sum((aux_design))
```
