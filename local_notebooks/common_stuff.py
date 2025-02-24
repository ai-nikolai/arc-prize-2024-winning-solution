# common configuration for training and evaluation
from arc_loader import *
from model_runner import *
from selection import *
from async_tools import *
import time

# paths
# base_path, running_on_kaggle = ('/kaggle', True) if os.path.exists('/kaggle') else ('.', False)
# tmp_dir = os.path.join(base_path, 'temp')

# OTHER PARAMS
MyFormatter = ArcFormatter_premix_3
perm_aug = 'rnd_all'
max_seq_length_train = 4224
mask_first = 0

# training & inference
train_epochs = 4
# NOTE THIS IS FOR MULTIGPU STUFF
multi_gpu_train = False
multi_gpu_random_split = False

max_seq_length_infer = 8192
prime_on_single_task = False
# infer_params = dict(min_prob=0.17, store=infer_temp_storage, use_turbo=True)

# scoring
use_aug_score = True
aug_score_params = dict(tp=True, rot=True, perm=perm_aug, shfl_ex=True, make_unique=True, max_len=max_seq_length_infer)
submission_select_algo = score_full_probmul_3 if use_aug_score else score_all_probsum


# DATASET RELATED (Same for all):
# base_path = os.path.join(base_path, "input")
current_file_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.join(current_file_path,"..","data")


arc_challenge_file = os.path.join(base_path, 'arc-prize-2024', 'arc-agi_evaluation_challenges.json')
arc_solutions_file = os.path.join(base_path, 'arc-prize-2024', 'arc-agi_evaluation_solutions.json')

# load datasets
arc_test_set = ArcDataset.from_file(self.arc_challenge_file)
if arc_test_set.is_fake: arc_test_set.load_replies(arc_solutions_file)


def prepare_run(model_path, load_lora=None, train=False, gpu=None, **kwargs):
    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"   ] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    model, tokenizer, formatter = prepare_model(  # base model configuration
        model=model_path,
        local_files_only=True,
        mode='unsloth_4bit',
        #shrink_embedding=8000,
        max_seq_length=max_seq_length_train,
        formatter=MyFormatter,
        peft=([dict(
            r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head'],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing=True,  # True or "unsloth" for very long context
            random_state=42,
            use_rslora=True,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )] if train or load_lora else []) + ([load_lora] if load_lora else []),
        **kwargs
    )
    
    if train and mask_first: formatter.collator_kwargs.update(mask_first_output=mask_first)

    return model, formatter

def prepare_dataset(formatter, train, gpu=None, augment=False):
    ds = arc_test_set
    if multi_gpu_train and gpu is not None:
        if multi_gpu_random_split:
            ds = ds.shuffled(seed=123)
            ds = ds.split_at_pos(len(ds.keys)//2)[gpu]
        else:
            ds = ds.sorted_by_len(formatter=formatter, name='input', max_of_transposed=True)
            assignment = ([0,1,1,0]*ds.length())[:ds.length()][::-1]
            ds = ds.change_keys((np.array(ds.keys)[np.array(assignment)==gpu]).tolist())
    if train:
        ds = ds.remove_replies()
        if augment:
            ds = ds.augment(tp=True, rot=True, perm=perm_aug, n=(2 if arc_test_set.is_fake else train_epochs), shfl_ex=True, shfl_keys=True)
        else:
            print("Note we are skipping augmentation")
        ds = ds.cut_to_len(formatter=formatter, name='text', max_len=max_seq_length_train, max_new_tokens=0)
        if arc_test_set.is_fake: ds = ds.sorted_by_len(formatter=formatter, name='text', reverse=True)
    else:
        ds = ds.sorted_by_len(formatter=formatter, name='input', max_of_transposed=True)
        ds = ds.split_multi_replies()
        if augment:
            ds = ds.augment(tp=True, rot=True, n=2, seed=42, perm=perm_aug, shfl_ex=True).interleave(ds.length())
        else:
            print("Note we are skipping augmentation")
        ds = ds.cut_to_len(formatter=formatter, name='input', max_len=max_seq_length_infer)
        if arc_test_set.is_fake: ds.keys = ds.keys[:128] #ds.keys[::-1][::5][::-1]
    return ds

class Trainer:

    def __init__(self, main_name="temp"):
        tmp_dir = os.path.join(current_file_path, main_name)
        self.model_temp_storage = os.path.join(tmp_dir, 'finetuned_model')
        self.infer_temp_storage = os.path.join(tmp_dir, 'inference_outputs')
        self.score_temp_storage = os.path.join(tmp_dir, 'inference_scoring')

        self.base_model = download_model('da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit', tmp_dir)

    def start_training(self,gpu, augment=False):
        base_model = self.base_model
        model_temp_storage=self.model_temp_storage 
        infer_temp_storage=self.infer_temp_storage
        score_temp_storage=self.score_temp_storage
        try:
            print(f"{'='*20}\n\nWe are running: {arc_test_set.is_fake}")
            storage_path = f'{model_temp_storage}_gpu{gpu}'
            if (gpu==0 or multi_gpu_train) and not os.path.exists(storage_path):
                with RemapCudaOOM():
                    model, formatter = prepare_run(base_model, train=True, gpu=gpu)
                    dataset = prepare_dataset(formatter, train=True, gpu=gpu if multi_gpu_train else None, augment=augment)
                    model, trainer_stats = training_run(
                        model, formatter, dataset, store=storage_path,
                        max_seq_length=max_seq_length_train,
                        grad_acc_fix=False,
                        train_args=dict(
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=2,
                            warmup_steps=100,
                            num_train_epochs=1,
                            max_steps=20 if arc_test_set.is_fake else -1,
                            learning_rate=1e-4,
                            embedding_learning_rate=1e-5,
                            logging_steps=10,
                            optim="adamw_8bit",
                            weight_decay=0.01,  # 0.01,
                            lr_scheduler_type='cosine',  # "linear", "cosine",
                            seed=42,
                            output_dir=os.path.join(tmp_dir, 'checkpoints'),
                            save_strategy="no",
                            report_to='none',
                        ),
                    )
                    mem_info()
        finally: os.makedirs(f'{storage_path}_done', exist_ok=True)

    def start_inference(self, gpu, augment=False):
        base_model = self.base_model
        model_temp_storage=self.model_temp_storage 
        infer_temp_storage=self.infer_temp_storage
        score_temp_storage=self.score_temp_storage
        
        storage_path = f'{model_temp_storage}_gpu{gpu if multi_gpu_train else 0}'
        while not os.path.exists(f'{storage_path}_done'): time.sleep(15)
        with RemapCudaOOM():
            model, formatter = prepare_run(storage_path, gpu=gpu)
            dataset = prepare_dataset(formatter, train=False, gpu=gpu, augment=augment)
            retrainer = None if not prime_on_single_task else Retrainer(
                n=32,
                aug_opts=dict(perm=perm_aug, shfl_ex=True),
                reload_state_dict=get_and_fix_peft_weights(storage_path),
                formatter=formatter,
                max_seq_length=max_seq_length_infer,
                grad_acc_fix=False,
                train_args=dict(
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    warmup_steps=4,
                    num_train_epochs=1,
                    learning_rate=1e-4,
                    embedding_learning_rate=0,
                    logging_steps=8,
                    optim="adamw_8bit",
                    weight_decay=0.00,  # 0.01,
                    lr_scheduler_type='constant',  # "linear", "cosine",
                    seed=42,
                    output_dir='tmp_output',
                    save_strategy='no',
                    report_to='none',
                ),
            )
            decoder = Decoder(formatter, arc_test_set.split_multi_replies(), n_guesses=2, prob_baseline=0.05)
            inference_run_v2(model, formatter, dataset, decoder, retrain=retrainer, **infer_params)
            if use_aug_score or arc_test_set.is_fake: decoder.calc_augmented_scores(model=model, store=score_temp_storage, **aug_score_params)
            mem_info()

class RemapCudaOOM:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_value, traceback):
        oom_errors = ["CUDA out of memory", "Make sure you have enough GPU RAM", "does not fit any GPU's remaining memory"]
        if exc_value and any(x in str(exc_value) for x in oom_errors):
            with open('submission.json', 'w') as f: f.write('cause submission scoring error')
