Overall Algorithm
=================

The overall estimation algorithm performs steps in the following order:
1. Unadjusted Design Weight
2. Design Weight
3. Calibration Weight

* Expansion Estimation is the calculation of the Design Weight using Unadjusted Design Weight and Birth/Death adjustment.
* Separate and Combined Ratio Estimation use the Unadjusted Design Weight to calculate Calibration Weight. (They run Expansion Estimation first.)

* Expansion and Separate Ratio Estimation calculate values by Strata. Combined Ration calculates the Calibration Weight by Calibration Group (Its Unadjusted Design Weight is still by Strata).

Unadjusted Design Weight
========================

Estimation calculates this first as it is used in all types of estimation. It uses the sample_marker column which can only contain a 0 or a 1 allowing it to easily work out the total population and sample population using a count and a sum.

The calculation is:
```
unadjusted_deight_weight = count(sample_marker)/sum(sample_marker)
```

Estimation runs a check to ensure `sample_marker` only contains a 1 or a 0 and will throw an Error if there is a violation of this requirement.

Design Weight
=============

Taking the calculated `unadjusted_design_weight` estimation uses birth/death adjustment to get the final Design Weight. The calcultion uses the sample population, number of deaths in the population and the birth/death adjustment parameter (`h_value`).

The calculation is:
```
design_weight = unadjusted_design_weight * (1 + (h_value * (sum(death_marker)/(sum(sample_marker) - sum(death_marker)))))
```

Similar to the `sample_marker` the `death_marker` can only contain 0 or 1. The `h_value` is usually a 1 or 0 but could possibly be any positive number.
The `death_marker` and `h_value` are optional parameters as a birth/death adjustment may not always be wanted. Leaving them blank will set a `h_value` of 0. This means the `undajusted_design_weight` is always 'adjusted' by a factor of 1, a.k.a. it isn't adjusted.

Calibration Weight
==================

Estimation uses the `unadjusted_design_weight` again and uses the Auxiliary value for the responders to turn it into a Calibration Weight.

```
calibration_weight = sum(auxiliary_value)/sum((unadjusted_design_weight * auxiliary_value))
```