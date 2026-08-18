'''
patterns labeling as explained in Section 4.2.4
'''

def assign_pattern(row_isna, blocks):
    is_a_pattern = ['+'.join(b['itemset']) for b in blocks if row_isna[b['itemset']].all()]
    if not is_a_pattern:
        return 'mp0' if not row_isna.any() else "mp_other"
    return ' & '.join(is_a_pattern)

def assign_patterns(df_features, model):
    return df_features.apply(
        lambda row: assign_pattern(row.isna(), model.blocks), axis=1
    )