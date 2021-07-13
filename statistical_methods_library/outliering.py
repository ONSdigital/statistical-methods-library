"""
This module performs outliering. Currently 1-sided Winsorisation is
implemented.
"""
group_cols = periodColumn + strataColumns

df.groupBy(group_cols).withColumn("ratio_sum_target_sum_aux", expr("sum(target_column * design_weight)/sum(auxiliary_column * design_weight)"))
join back onto origional_df on group_cols
winsorisation_value = ratio_sum_target_sum_aux * auxiliary_column

k_value = winsorisation_value + (l_value/((design_weight * calibration_weight) -1))

when target_column > k_value:
	modified_target = (target_column/(design_weight*calibration_weight)) + (k_value - (k_value/(design_weight*calibration_weight)))
else:
	modified_target = target_column

o_weight = modified_target/target_column
