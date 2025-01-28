import json
import numpy as np
import hashlib
import os, sys
from tqdm import tqdm
from glob import glob
import itertools
import random

def cut_at_token(output, token_id):
    eos_positions = (output==token_id).nonzero()[0]
    return output[:eos_positions[0]] if len(eos_positions) else output

def shuffled(data_list):
    return np.random.permutation(data_list).tolist()

def permute_mod(a, descriptor, invert=False):
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation)==list(range(10))
    a = np.asarray(a)
    if a.ndim==3:
        if not invert: permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        assert a.ndim==2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a

def permute_rnd_col_(query):
    permutation = [0]+(1+np.random.permutation(9)).tolist()
    return 'permute' + ''.join(map(str, permutation))

def permute_rnd_all_(query):
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))

def permute_cnt_col_(query):
    elements, frequency = np.unique(np.concatenate([list(range(10))]+[np.array(x['input']).ravel() for x in query['train']]), return_counts=True)
    permutation = [0]+sorted(np.random.permutation(9)+1, key=lambda i: frequency[i], reverse=True)  # randomness as tie breaker
    return 'permute' + ''.join(map(str, permutation))

def permute_cnt_all_(query):
    elements, frequency = np.unique(np.concatenate([list(range(10))]+[np.array(x['input']).ravel() for x in query['train']]), return_counts=True)
    permutation = sorted(np.random.permutation(10), key=lambda i: frequency[i], reverse=True)  # randomness as tie breaker
    return 'permute' + ''.join(map(str, permutation))

permute_rnd_col = (permute_mod, permute_rnd_col_)
permute_rnd_all = (permute_mod, permute_rnd_all_)
permute_cnt_col = (permute_mod, permute_cnt_col_)
permute_cnt_all = (permute_mod, permute_cnt_all_)
permute_None = (np.copy, None)

class ArcDataset(object):
    @staticmethod
    def forward_mod(a, key, use_perm=True, is_output=True):
        if a is None: return a
        for op in key.split('.')[1:]:
            if op.startswith('I'):
                if is_output: continue
                op = op[1:]
            if   op=='rot90':              a = np.rot90(a)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('fix'):     a = a
            elif op.startswith('ice'):     a = a  # for adding icecuber solutions
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True, is_output=True):
        if a is None: return a
        for op in key.split('.')[1:][::-1]:
            if op.startswith('I'):
                if is_output: continue
                op = op[1:]
            if   op=='rot90':              a = np.rot90(np.rot90(np.rot90(a)))
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('fix'):     a = a
            elif op.startswith('ice'):     a = a  # for adding icecuber solutions
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies={}, keys=None, is_orig=False, is_fake=False):
        if keys is not None: keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.is_fake = is_fake
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.faulty = {}
        self.transposed_dataset = None

    @classmethod
    def empty(cls):
        return cls(queries={}, replies={}, keys=[])

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_fake=self.is_fake, is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, queries_file):
        print(f"*** Load challanges from '{queries_file}'...")
        with open(queries_file) as f: queries = f.read()
        is_fake = hashlib.md5(queries.encode('utf-8')).hexdigest().lower()=='a6b7dac3cab03abf2eb333e16610d6dc'
        if is_fake: print("*** -> Fake test set detected, setting flag 'is_fake' to True.")
        return cls(
            queries=json.loads(queries),
            is_fake=is_fake,
            is_orig=True,
        )

    def load_replies(self, replies_file):
        print(f"*** Load solutions from '{replies_file}'...")
        with open(replies_file) as f: replies = f.read()
        replies_parsed = json.loads(replies)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self):
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )

    def move_test_to_train(self):
        new_queries = {k: {'train': self.queries[k]['train'] + [{**t, 'output': self.replies[k][i]} for i, t in enumerate(self.queries[k]['test'])], 'test': []} for k in self.keys}
        return self.__class__(queries=new_queries, keys=[k for k in self.keys])

    def last_train_ex_for_test(self):
        assert not self.replies
        new_queries = {k: {'train': self.queries[k]['train'][:-1], 'test': [{'input': self.queries[k]['train'][-1]['input']}]} for k in self.keys}
        new_replies = {k: [self.queries[k]['train'][-1]['output']] for k in self.keys}
        return self.__class__(queries=new_queries, replies=new_replies, keys=[k for k in self.keys])

    def length(self):
        return len(self.keys)

    def shuffled(self, seed=None):
        if seed is not None: np.random.seed(seed)
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    def sorted(self, **kwargs):
        return self.__class__(queries=self.queries, replies=self.replies, keys=sorted(self.keys, **kwargs))

    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys   =[k    for d in datasets for k    in d.keys           ],
        )

    def sort_ex_by_input_size(self, seed=42, reverse=False):
        np.random.seed(seed)
        sort_key = lambda ex: np.prod(np.shape(ex['input']))
        new_queries = {k2: {k: (sorted(np.random.permutation(np.array(v, dtype=object)), key=sort_key, reverse=reverse) if k=='train' else v) for k, v in v2.items()} for k2, v2 in self.queries.items()}
        return self.__class__(queries=new_queries, replies=self.replies, keys=[k for k in self.keys])

    def interleave(self, block_size, num_gpus=None):
        keys = np.reshape(self.keys, (-1, block_size)).T
        if num_gpus is None: return self.change_keys(keys.ravel().tolist())
        ret, num_gpus = (None, num_gpus) if isinstance(num_gpus, int) else num_gpus
        keys = np.concatenate([keys, np.full((-keys.shape[0]%num_gpus, keys.shape[1]), None)])
        keys = np.reshape(keys, (keys.shape[0]//num_gpus, num_gpus, -1)).swapaxes(0, 1).reshape(num_gpus, -1)
        new_datasets = [self.change_keys(gpu_keys.tolist()) for gpu_keys in keys]
        return new_datasets if ret is None else new_datasets[ret]

    def remove(self, *datasets):
        remove_keys = {k for d in datasets for k in d.keys}
        new_keys = [k for k in self.keys if k not in remove_keys]
        return self.change_keys(new_keys)

    def keep_key_startswith(self, key_start):
        new_keys = [k for k in self.keys if k.startswith(key_start)]
        return self.change_keys(new_keys)

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        queries = {}
        replies = {}
        keys    = []
        for k0 in self.keys:
            desc = (('copy{i}' if mod_func is np.copy else mod_func.__name__) if descriptor is None else descriptor if isinstance(descriptor, str) else descriptor(self.queries[k0])).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a, desc) if t=='input' or not inputs_only else a) for t, a in x.items()} for x in e] for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        ret = self.__class__(queries=queries, replies=replies, keys=keys)
        return ret

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None: stack = mod_func.__name__.startswith('rot')
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter):
        assert formatter.out2_token is None or key in self.replies
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'], i=len(self.queries[key]['train']))
        reply = formatter.fmt_reply(self.replies[key], self.faulty.get(key)) if key in self.replies else ''
        text = train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, train=train, query=query, reply=reply, input=train+query, text=text)

    def as_list(self, formatter):
        return [self.get(key, formatter) for key in self.keys]

    def as_dataset(self):
        from datasets import Dataset
        return Dataset.from_list([{'key': k, 'query': self.queries[k], 'reply': self.replies[k]} for k in self.keys])

    def get_length(self, key, formatter, name, max_of_transposed=False):
        if formatter is None:
            if   name=='input': return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name=='reply': return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else: assert False
        else:
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None: self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer(ds.get(key, formatter=formatter)[name])['input_ids']) for ds in datasets)

    def get_lengths(self, formatter, name, max_of_transposed=False):
        return {key: self.get_length(key, formatter=formatter, name=name, max_of_transposed=max_of_transposed) for key in self.keys}

    def sorted_by_len(self, reverse=False, **kwargs):
        new_keys = [key for _, key in sorted([(v, k) for k, v in self.get_lengths(**kwargs).items()], reverse=reverse)]
        return self.change_keys(new_keys)

    def filter_by_len(self, min_len=0, max_len=float('inf'), **kwargs):
        new_keys = [k for k, v in self.get_lengths(**kwargs).items() if min_len<=v<=max_len]
        return self.change_keys(new_keys)

    def cut_to_query_count(self, max_count, from_end=False):
        new_queries = {}
        for k in self.keys:
            new_queries[k] = q = self.queries[k]
            while len(q['train'])>max_count: q['train'] = q['train'][:-1] if from_end else q['train'][1:]
        return self.__class__(queries=new_queries, replies=self.replies, keys=[k for k in self.keys])

    def cut_to_len(self, formatter, name, max_len, max_new_tokens='auto', from_end=False, quiet=False, **kwargs):
        if max_new_tokens:
            if max_new_tokens=='auto': max_new_tokens = formatter.max_new_tokens()
            max_len_old, max_len = max_len, max_len - max_new_tokens
            if not quiet: print(f'*** Reducing task size to max. {max_len_old} tokens ({max_len} input + {max_new_tokens} generated)...')
        elif not quiet: print(f'*** Reducing task size to max. {max_len} tokens...')
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in (self.keys if quiet else tqdm(self.keys, file=sys.stdout)):
            reply = temp_ds.replies.get(key)
            while max_len<temp_ds.get_length(key, formatter=formatter, name=name, **kwargs):
                query = temp_ds.queries[key]
                if not key.split('.')[-1].startswith('ex'): key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k=='train' else v) for k, v in query.items()}
                if reply is not None: temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None: new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)

    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None: p = p[:keep_max]
            new_key = f'{key}.ex' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='train' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def shuffle_rp(self, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['test'])
            p = np.random.permutation(n)
            if keep_max is not None: p = p[:keep_max]
            new_key = f'{key}.rp' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='test' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = np.array(self.replies[key], dtype=object)[p].tolist()
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def append_to_keys(self, test):
        return self.change_keys([f'{k}{text}' for k in self.keys])

    def random_select(self, n):
        keys = np.array(self.keys).reshape(n, -1).T
        choice = np.random.randint(0, n, size=[len(keys)])
        return self.change_keys(keys[np.arange(len(keys)), choice])

    def augment(self, tp=False, rot=False, n=1, perm=None, perm_append=False, shfl_keys=False, shfl_ex=False, seed=None, quiet=False, inputs_only=False):
        if not quiet: print(f"*** Augment dataset{' (inputs only)' if inputs_only else ''}...")
        np.random.seed(seed)
        d = self
        if tp: d = d.mod(np.transpose, keep=True, inputs_only=inputs_only)
        if tp=='rand': d = d.random_select(n=2)
        if rot: d = d.mod(np.rot90, n=3, keep=True, inputs_only=inputs_only)
        if rot=='rand': d = d.random_select(n=4)
        if perm is None and n<=1: d = d.shuffled() if shfl_keys else d
        else: d = d.mod(*([np.copy] if perm is None else globals()[f"permute_{perm}"]), n=n, shuffle=shfl_keys, keep=perm_append, inputs_only=inputs_only)
        np.random.seed(seed)
        if shfl_ex: d = d.shuffle_ex()
        return d

    def remove_replies(self):
        return self.__class__(queries=self.queries, replies={}, keys=[k for k in self.keys])

    def split_at_pos(self, pos, random_seed=None):
        keys = self.keys
        if random_seed is not None:
            np.random.seed(random_seed)
            keys = np.random.permutation(keys)
        if isinstance(pos, float): pos = int(pos * len(self.keys) + 0.5)
        keys_split = [keys[:pos], keys[pos:]]
        return tuple(self.change_keys(new_keys, keep_flags=True) for new_keys in keys_split)

    def get_submission(self, results=None):
        assert self.is_orig==True, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(self.queries[k]['test']))] for k in self.keys}
        if results is not None: self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        print(f'*** Generating submission for {len(results)} outputs...')
        for k, v in results.items():
            base_id, base_nr = k.split('_')
            target_dict = submission[base_id][int(base_nr)]
            for i, g in enumerate(v[:len(target_dict)]):
                target_dict[f'attempt_{i+1}'] = g.tolist()

    def validate_submission(self, submission):
        assert self.is_orig==True, 'Must be run on original dataset.'
        score = 0
        for k, v in self.replies.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score
def get_class_MyDataCollator(cache=[]):
    if not cache:
        from trl import DataCollatorForCompletionOnlyLM
        class MyDataCollator(DataCollatorForCompletionOnlyLM):
            def setup(self, out2_token_id=None, fault_token_id=None, fault_freq=0, sample_tries=8, mask_first_output=False):
                self.out2_token_id = out2_token_id
                self.fault_token_id = fault_token_id
                self.fault_freq = fault_freq
                self.sample_tries = sample_tries
                self.mask_first_output = mask_first_output
                return self

            def torch_call(self, examples):
                batch = super().torch_call(examples)
                if self.out2_token_id is not None:
                    assert not self.fault_freq
                    for i in range(len(batch['input_ids'])):
                        end_pos = ((batch['labels'][i] != -100              ).nonzero().max()).item() + 1
                        mid_pos = ((batch['labels'][i] == self.out2_token_id).nonzero().max()).item() + 1
                        beg_pos = mid_pos - (end_pos - mid_pos)
                        batch['labels'][i][beg_pos:mid_pos] = batch['labels'][i][mid_pos:end_pos]
                elif self.fault_freq:
                    for i in range(len(batch['input_ids'])):
                        end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                        if not isinstance(self.fault_freq, float):
                            eos_token_id = batch['labels'][i][end_pos - 1]
                            num_examples = (batch['labels'][i] == eos_token_id).sum().item() - 1
                            fault_freq = self.fault_freq[num_examples]
                        else: fault_freq = self.fault_freq
                        if random.random() < fault_freq:
                            beg_pos = ((batch['labels'][i][:end_pos]==-100).nonzero().max()).item() + 1
                            fault_pos = random.randint(beg_pos, end_pos-2)
                            fault_tok = batch['labels'][i][fault_pos].item()
                            for t in range(self.sample_tries):
                                new_tok = batch['labels'][i][random.randint(beg_pos, end_pos-2)].item()
                                if fault_tok!=new_tok:
                                    batch['input_ids'][i][fault_pos] = new_tok
                                    batch['labels'][i][fault_pos+1:end_pos] = self.fault_token_id
                                    break
                for i in range(len(batch['labels'])):
                    for _ in range(self.mask_first_output):
                        beg_pos = ((batch['labels'][i] != -100).nonzero().min()).item()
                        mid_pos = ((batch['labels'][i][beg_pos:] == -100).nonzero().min()).item() + beg_pos
                        end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                        if mid_pos<end_pos: batch['labels'][i][beg_pos:mid_pos] = -100
                return batch
        cache.append(MyDataCollator)
    return cache[0]

class ArcFormatter(object):
    def __init__(self, inp_prefix, out_prefix, arr_sep, out2_use=False, out2_token=None, arr_beg='', arr_end='', pretext='', pre_out=None, exa_sep='', exa_end='', qry_prefix=None, rpl_prefix=None, rpl_sep=None, dec_sep=None, min_wid=0, min_pad='', pretext_corpus_split='', masking=0, tokenizer=None, collator_kwargs={}, repeat_input_aug=None, repeat_input_pre=None):
        self.tokenizer = tokenizer
        self.inp_prefix = inp_prefix
        self.out_prefix = out_prefix
        self.out2_token = out2_token
        self.out2_use = out2_use
        assert not out2_use or out2_token is not None
        assert not out2_use or masking in [1, 2]
        assert masking!=2 or out2_use or rpl_prefix is not None
        self.qry_prefix = qry_prefix if qry_prefix is not None else inp_prefix
        self.rpl_prefix = rpl_prefix if rpl_prefix is not None else out_prefix
        self.rpl_sep = rpl_sep if rpl_sep is not None else self.rpl_prefix
        self.arr_sep = arr_sep
        self.arr_beg = arr_beg
        self.arr_end = arr_end
        self.pretext = pretext
        self.pre_out = pre_out
        self.pre_out_empty = ['']*99
        self.pretext_corpus_split = pretext_corpus_split
        self.exa_sep = exa_sep
        self.exa_end = exa_end
        self.dec_sep = arr_sep if dec_sep is None else dec_sep
        self.min_wid = min_wid
        self.min_pad = min_pad
        self.masking = masking
        self.collator_kwargs = collator_kwargs
        self.repeat_input_aug = repeat_input_aug
        self.repeat_input_pre = repeat_input_pre

    def fmt_array(self, array):
        return self.arr_beg + self.arr_sep.join(str(row).replace(' ', '').replace(',', '').replace('[', '').replace(']', '')+self.min_pad*max(0, self.min_wid-len(row)) for row in array) + self.arr_end

    def get_pre_out(self, pretext_split):
        if self.pre_out is None: return self.pre_out_empty
        if pretext_split: return [self.pretext_corpus_split.join(list(p) + ['']) for p in self.pre_out]
        return self.pre_out

    def fmt_train(self, train, last_is_challenge=False, pretext_split=False):
        po = self.get_pre_out(pretext_split=pretext_split)
        ex = [(f"{self.fmt_query([x], i, pretext_split=pretext_split)}{self.fmt_reply([x['output']])}" if last_is_challenge and i+1==len(train) else
               f"{self.inp_prefix}{self.fmt_array(x['input'])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{self.out_prefix}{self.fmt_array(x['output'])}") for i, x in enumerate(train)]
        pre = self.pretext_corpus_split.join(list(self.pretext)+['']) if pretext_split else self.pretext
        end = '' if last_is_challenge else (self.exa_end + self.tokenizer.eos_token)
        return pre + (self.exa_end + self.tokenizer.eos_token + self.exa_sep).join(ex) + end

    def fmt_query(self, query, i, pretext_split=False):
        po = self.get_pre_out(pretext_split=pretext_split)
        return ''.join(f"{self.qry_prefix}{self.fmt_array(x['input'])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{self.rpl_prefix}" for x in query[:1])

    def repeat_input(self, x, no_aug=False):
        if self.repeat_input_aug is None: return ''
        return f"{self.repeat_input_pre}{self.fmt_array(((lambda x: x) if no_aug else self.repeat_input_aug)(x['input']))}"

    def fmt_reply(self, reply, fault=None):
        ids = self.fmt_array(reply[0]) + self.exa_end + self.tokenizer.eos_token
        if self.out2_use:
            if fault is None: fault = reply
            ids = self.fmt_array(fault[0]) + self.exa_end + self.out2_token + ids
        return ids

    def quick_test(self, decoded, done):
        sp = decoded.split(self.tokenizer.eos_token)[0].split(self.dec_sep)
        sl = len(sp[0])
        is_prefix = sl>0 and len(sp[-1])<=sl and (len(sp)==1 or len(sp[-2])==sl) and all(x.isdigit() for x in sp[-1])
        return is_prefix and (not done or len(sp[-1])==0 or len(sp[-1])==sl)

    @staticmethod
    def is_valid_solution(guess):
        return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)

    def max_new_tokens(self, safety_margin=1):
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokenized = self.tokenizer(self.fmt_reply([max_sized_reply]))['input_ids']
        max_new_tokens = len(tokenized)
        if tokenized[0]==self.tokenizer.bos_token_id: max_new_tokens -= 1
        return max_new_tokens + safety_margin

    def de_tokenize(self, tokens, scores=None):
        import torch
        tokens_cut = cut_at_token(tokens, self.tokenizer.eos_token_id)
        de_tokenized = self.tokenizer.batch_decode([tokens_cut])[0]
        score_val = None
        if scores is not None:
            tokens_with_eos = tokens[:len(tokens_cut)+1]
            score_val = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1).numpy().copy()[np.arange(len(tokens_with_eos)), tokens_with_eos].sum()
            number_token_ids = [self.tokenizer.vocab[k] for k in map(str, range(10))]
            fault_token_id = self.collator_kwargs.get('fault_token_id')
            if fault_token_id is not None: number_token_ids.append(fault_token_id)
            number_token_ids = np.array(number_token_ids)
            number_positions = (tokens_cut[..., np.newaxis] == number_token_ids).any(-1)
            scores = scores[:len(tokens_cut), number_token_ids][number_positions]
            scores = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1)[:, :10].numpy().copy()
        return max(len(tokens)+1, len(tokens_cut)), score_val, de_tokenized, scores

    def decode_to_array_single(self, text, score=None, limit_rows=30):
        try:
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in text.split(self.dec_sep)] if len(row)]
            if limit_rows and len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
                limited = True
            else: limited = False
            decoded = np.array(by_rows, dtype=int)
            if self.is_valid_solution(decoded):
                try:
                    assert score is not None
                    decoded_flat = decoded.ravel()
                    if limited: score = score[:len(decoded_flat)]
                    score_all = score.reshape(decoded.shape + score.shape[1:])
                    score_result = score[range(len(decoded_flat)), decoded_flat]
                    score_reshaped = score_result.reshape(decoded.shape)
                    score_cum_reshaped = score_result.cumsum().reshape(score_reshaped.shape)
                    score_all_cum = score_cum_reshaped[..., np.newaxis] - score_reshaped[..., np.newaxis] + score_all
                except: score_reshaped = score_cum_reshaped = np.full(decoded.shape, -float('inf'))
                return {'output': decoded, 'score': score_reshaped, 'score_cum': score_cum_reshaped, 'score_all': score_all, 'score_all_cum': score_all_cum}
        except: pass
        return {}

    def decode_to_array(self, text, score=None, limit_rows=30):
        if not self.out2_use: text, score = [text], [score]
        else:
            text = text.split(self.out2_token)
            if score is None: score = [None]*len(text)
            else:
                lengths = np.cumsum([len(list(filter(str.isdigit, t))) for t in text])
                score = [score[s:e] for s, e in zip([0]+lengths[:-1].tolist(), lengths)]
        return [self.decode_to_array_single(t, s) for t, s in zip(text, score)]

    def get_corpus(self):
        try:
            old_min_wid, self.min_wid = self.min_wid, min(self.min_wid, 2)
            return self.fmt_train([{'input': [[i] for i in range(10)], 'output': [[i] for i in range(10)]}]*3, last_is_challenge=True, pretext_split=True)
        finally: self.min_wid = old_min_wid

    def get_data_collator(self):
        if not self.masking: return None
        from transformers import DataCollatorForLanguageModeling
        collator_params = dict(tokenizer=self.tokenizer, mlm=False)
        pass_out2_token = self.tokenizer.vocab[self.out2_token] if self.out2_use and self.masking==1 else None
        if self.masking:
            assert not self.collator_kwargs.get('mask_first_output') or self.masking==1
            data_collator = get_class_MyDataCollator()(
                **collator_params,
                instruction_template=[self.inp_prefix, self.tokenizer.bos_token][self.masking - 1],
                response_template=[self.out_prefix, (self.out2_token if self.out2_use else self.rpl_sep)][self.masking - 1],
            ).setup(out2_token_id=pass_out2_token, **self.collator_kwargs)
        else:
            assert not self.collator_kwargs, 'only supported with masking on'
            data_collator = DataCollatorForLanguageModeling(**collator_params)
        return data_collator

    def get_output_token_ids(self):
        assert not self.out2_use
        num_tokens = [self.tokenizer.vocab[str(i)] for i in range(10)]
        sep_tokens = [tok for txt in [self.arr_beg, self.arr_sep, self.arr_end, self.exa_sep] if txt for tok in self.tokenizer(txt)['input_ids'][1:]]
        sep_tokens.append(self.tokenizer.eos_token_id)
        return num_tokens + sorted(set(sep_tokens))

ArcFormatter_pretext2 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZ', pretext_corpus_split='\n', **kwargs)
ArcFormatter_pretext3 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz', pretext_corpus_split='\n', **kwargs)
ArcFormatter_premix_2 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZ', pre_out=['+/-=']*99, pretext_corpus_split='\n', **kwargs)
ArcFormatter_premix_3 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz', pre_out=['+/-=']*99, pretext_corpus_split='\n', **kwargs)

available_formatters = dict(
    ArcFormatter_pretext2=ArcFormatter_pretext2,
    ArcFormatter_pretext3=ArcFormatter_pretext3,
    ArcFormatter_premix_2=ArcFormatter_premix_2,
    ArcFormatter_premix_3=ArcFormatter_premix_3,
)