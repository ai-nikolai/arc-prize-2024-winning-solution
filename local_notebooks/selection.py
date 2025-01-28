import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def make_unique(guess_list, indices=None):
    used = set()
    out = []
    out_ind = []
    for i, g in enumerate(guess_list):
        h = hashable(g)
        if h not in used:
            used.add(h)
            out.append(np.array(g))
            if indices is not None: out_ind.append(indices[i])
    return out if indices is None else (out, out_ind)

def first_only(guesses):
    return [g['output'] for g in guesses.values()][:1]

def keep_order(guesses):
    return [g['output'] for g in guesses.values()]

def keep_order_unique(guesses):
    return make_unique(keep_order(guesses))

def get_best_shape_by_score(guess_list, getter, once_per_result=True):
    seen_outputs = set()
    shape_scores = {}
    for i, g in enumerate(guess_list):
        shape = tuple(g['output'].shape)
        scores = shape_scores[shape] = shape_scores.get(shape, [[], []])
        scores[1].append(i)
        h = hashable(g['output'])
        if h in seen_outputs: continue
        if once_per_result: seen_outputs.add(h)
        scores[0].append(g)
    shape_scores = [(getter(scores), shape, indices) for shape, (scores, indices) in shape_scores.items()]
    shape_scores = sorted(shape_scores, key=(lambda x: x[0]), reverse=True)
    return shape_scores[0]

def score_sum(guesses, getter, shape_getter=None, prefer_common_shape=True):
    if shape_getter is None: shape_getter = getter
    guess_list = list(guesses.values())
    common_shape_indices = set(get_best_shape_by_score(guess_list, shape_getter)[2]) if prefer_common_shape else []
    scores = {}
    for i, g in enumerate(guess_list):
        h = hashable(g['output'])
        x = scores[h] = scores.get(h, [i in common_shape_indices, [], g['output']])
        x[1].append(g)
    scores = [(cs, getter(sc), o) for cs, sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[:2]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

getter_all_probsum = lambda guesses: sum(np.exp(g['score_val']) for g in guesses)
def score_all_probsum(guesses): return score_sum(guesses, getter_all_probsum)

def getter_full_probmul(p):
    def _getter(guesses, baseline=p):
        inf_score = sum([g['score_val']+baseline for g in guesses])
        aug_score = np.mean([sum(s+baseline for s in g['score_multi_nl']) for g in guesses])
        return inf_score + aug_score
    return _getter

def score_full_probmul_3(guesses): return score_sum(guesses, getter_full_probmul(3), prefer_common_shape=False)

selection_algorithms = [
    first_only,
    keep_order,
    keep_order_unique,
    score_all_probsum,
    score_full_probmul_3,
]