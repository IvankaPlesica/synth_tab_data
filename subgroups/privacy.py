'''
on top of anonymeter, singling out
'''

def attacked_rows(control_df, successful_queries):
    attacked = set()
    for query in successful_queries:
        try:
            matched = control_df.query(query, engine='python')
        except Exception:
            continue
        if len(matched) == 1:
            attacked.add(matched.index[0])
        
    return attacked
        