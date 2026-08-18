'''
co-missingness block mining as described in RQ1
uses Apriori frequent itemset mining over every row's missingness indicator vector, filtered by any_confidence
'''

from efficient_apriori import apriori

def detect_relevant_blocks(itemsets, min_any_confidence):
    '''evaluate on every frequent itemset
    if a subset of a larger itemset is stronger, smaller is kept
    '''

    all_sets = [(frozenset(items), level[items].itemset_count)
                for level in itemsets.values() for items in level
                ]
    
    results = []
    for itemset, count in all_sets:
        if len(itemset) < 2: 
            continue

        singleton_counts = [itemsets[1][(col,)].itemset_count
                            for col in itemset
                            if (col,) in itemsets.get(1,{})
                            ]
        if(len(singleton_counts) < len(itemset)):
            continue

        weakest = min(singleton_counts)
        strongest = max(singleton_counts)
        any_confidence = count/weakest
        all_confidence = count/strongest

        results.append({
            'itemset': sorted(itemset),
            'itemset_frozenset': itemset,
            'count': count,
            'any_confidence': round(any_confidence,3),
            'all_confidence': round(all_confidence,3),
            'cross_support': round(all_confidence/any_confidence,3),
            'is_relevant_block': any_confidence >= min_any_confidence,
            })

    results.sort(key=lambda r: -r['any_confidence'])

    # added later, choosing subsets that are stronger than its relevant superset

    relevant = [r for r in results if r['is_relevant_block']]
    relevant_sets = [r['itemset_frozenset'] for r in relevant]

    final = []
    for r in relevant:
        is_subset = any(r['itemset_frozenset'] < other
                        for other in relevant_sets
                        if other != r['itemset_frozenset']
                        )
        if not is_subset:
            final.append(r)

    for r in final:
        del r['itemset_frozenset']
    
    return final


class MissingnessModel:
    
    @staticmethod
    def missing_matrix(df):
        cols = [c for c in df.columns if df[c].isnull().any()]
        return df[cols].isnull()
    
    def fit(self, df, min_support, min_any_confidence):
        self.min_support = min_support
        self.min_any_confidence = min_any_confidence

        M = self.missing_matrix(df)
        self.columns = list(M.columns)
        self.base_rates = M.mean().to_dict()
        self.n_rows = len(M)

        cols = M.columns.to_numpy()
        transactions = [tuple(cols[row]) for row in M.to_numpy()]
        itemsets, _ = apriori(transactions, min_support=min_support, output_transaction_ids=True)
        all_blocks = detect_relevant_blocks(itemsets, min_any_confidence)

        self.blocks = []
        c_cols = set() #what's already used in blocks
        for b in all_blocks:
            if not b['is_relevant_block']: #skip if not relevant
                continue
            if c_cols & set(b['itemset']): # skip if used
                continue
            self.blocks.append(b)
            c_cols.update(b['itemset']) # return only those that share no columns

        self.block_conditionals = {}
        for b in self.blocks:
            key = tuple(b['itemset'])
            block_missing = M[b['itemset']].all(axis=1)
            self.block_conditionals[key] = {
                col: block_missing[M[col]].mean() for col in b['itemset']
            }
        return self 

    def summary(self):
        print(f"Relevant blocks ({len(self.blocks)}):")
        for b in self.blocks:
            print(f"  {b['itemset']}  any_confidence={b['any_confidence']:.3f}")
