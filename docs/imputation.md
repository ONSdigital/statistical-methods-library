Link Calculation
================

We can calculate the links on a per-strata level in one direction then use
the reciprocal of the previous period's to calculate the other direction's
links. Links can then be joined onto the rest of the data by strata such
that each contributor will have its own links. In the event that the link
columns are provided these steps can be skipped since the link columns can
be selected as the correctly named columns for imputation.

The link calculation is:
```
forward_link(current_period) = sum(responders_in_both(current_period))
    /sum(responders_in_both(previous_period)

backward_link(current_period) = 1/forward_link(next_period)

link(current_period) = forward_link(current_period) if direction is forward
    or backward_link(current_period)
```

Where `responders_in_both(period)` is a function which returns the values for
the specified period for responders in both the current and previous
periods. If there are no matching responders this will be null. A responder
is a contributor for whom the value is a response rather than imputed or
constructed. If the denominator = 0 (null return from contributors_in_both
or 0 sum) then the denominator = 1. If there are no matching contributors
then `link(current_period)` returns 1.

Imputation
==========

Imputation only needs to work with current and previous periods since the
data can be reordered to do either direction. It simplifies to the process
of determining whether we have a value in the output column for the previous
period and a link for the current period and multiplying both to give the
current period's value. No knowledge of strata etc is required.

The imputation calculation is:
```
impute(current_period) = contributor(previous_period)*link(current_period)
```

Where `contributor(previous_period)` is the contributor's value for that
period. If this value is null then `impute(current_period)` returns null. If
the contributor is not present in the sample in the previous period then
`impute(current_period)` returns null. This is also the case if the target
value for the contributor is null in the previous period. This behaviour
means values for a contributor on either side of a gap (either because the
contributor was dropped from the sample or has no responses or constructed
values) do not interact.

Forward Imputation
------------------

For forward imputation the data is ordered by period in ascending order. The
markers for forward imputation are "FIR" and "FIC" where "FIR" means
forward imputation from response and "FIC" means forward imputation from
construction.

Backward Imputation
-------------------

In backward imputation the data is ordered by period in descending order.
The marker for backward imputation is "BI". Of note is that backward
imputation takes priority over both constructed values and forward
imputation from constructed values. As such, if any of these values are
present in the block for which backward imputation is being performed they
will be replaced as if they were non-responders. In addition, backward
imputation from construction is not valid.

Construction
============

The construction calculation is:
```
construction(current_period) = auxiliary(current_period)
    *sum(responders(current_period))
    /sum(auxiliaries_for_responders(current_period))
```

Where `responders(current_period)` returns the values for the responders in
the current period, `auxiliaries_for_responders(current_period)` returns the
auxiliary values for the responders in the current period and
`auxiliary(current_period)` is the auxiliary value for the contributor being
constructed for the current period. If the denominator is 0 then the
denominator defaults to 1. If any values in the auxiliary column are null a
runtime error is raised. The marker for constructed values is "C".

A construction filter can also be provided. In this case construction is
only performed when the filter returns true. Otherwise construction is
skipped for the contributor and period.

Overall Algorithm
=================

The overall imputation algorithm performs steps in the following order:
1. Link calculation
2. Forward imputation from response
3. Backward imputation
4. Construction
5. Forward imputation from construction

* If, at any point, the output column no longer contains null values, the
function returns. 

* If nulls are still present at the end of the process, a
runtime error is raised.

TODO
====

The above specification contains a case where imputation is impossible.
Specifically, when the construction filter is provided and construction is
required nulls will remain in the output column. This is a problem since
imputation must not output null values. As such a suitable behaviour must be
defined for this case.
