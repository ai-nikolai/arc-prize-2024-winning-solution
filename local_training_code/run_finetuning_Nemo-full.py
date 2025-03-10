# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer
from model_tools import load_peft_state, merge_peft_into_base

def clean_model_name(name):
    # Remove any path info, keep only last part
    name = name.split('/')[-1]
    # Replace punctuation with underscore
    import re
    name = re.sub(r'[^\w\s-]', '_', name)
    # Replace consecutive dashes/underscores with single dash
    name = re.sub(r'[-_]+', '-', name)
    return name.strip('-_')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='nvidia/Mistral-NeMo-Minitron-8B-Base')
    parser.add_argument("--save_name", default="IKCL")
    parser.add_argument("--remove_tokens", action="store_true")
    parser.add_argument("--augmentation_level", default="none")

    args = parser.parse_args()


    current_file_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(current_file_path,"..","data")

    # input paths
    base_model = args.model  # auto-downloaded from huggingface.co
    arc_data_path = os.path.join(base_path, 'arc-prize-2024')  # as on kaggle arc prize 2024
    re_arc_path = os.path.join(base_path, 're_arc')  # https://github.com/michaelhodel/re-arc
    neoneye_path = os.path.join(base_path, 'arc-dataset-collection-main')  # https://github.com/neoneye/arc-dataset-collection

    # output paths
    rm_tokens_str = "_rm-token" if args.remove_tokens else ""
    aug_str = f"_aug-{args.augmentation_level}"
    new_model_name = clean_model_name(args.model+"_"+args.save_name+aug_str+rm_tokens_str)
    print("="*20)
    print(f"The model name is:{new_model_name}")
    save_model_path = os.path.join('pretrained_models', new_model_name)

    for action in ['train', 'merge']:
        # continue if task already accomplished
        if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
            continue
        if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
            continue

        # load base model & reduce embedding size
        model = tokenizer = None  # free memory
        model, tokenizer = load_unsloth_4bit(base_model)
        
        # Now choosing to remove.
        if args.remove_tokens:
            keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
            keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

        # set formatting options
        fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n' + tokenizer.eos_token,
            lines_sep='\n',
            max_tokens=8192,
        )

        # create lora model
        lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']
        model = FastLanguageModel.get_peft_model(
            model=model,
            target_modules=lora_layers,
            r=256,
            lora_alpha=24,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            use_rslora=True,
            loftq_config=None,
        )

        if action == 'train':
            print("="*40)
            print("TRAIN LOOP")
            # load training data
            print("-"*20)
            print("Loading Data...")
            arc_eval_set = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi-fixed_evaluation_challenges_v1.json'))
            arc_eval_set = arc_eval_set.load_solutions(os.path.join(arc_data_path, 'arc-agi-fixed-evaluation_solutions_v1.json'))
            concept_arc = ArcDataset.load_from_neoneye(os.path.join(neoneye_path, 'dataset', 'ConceptARC'))
            mix_datasets = {
                'arceval': arc_eval_set.move_test_to_train().repeat(128),
                'concept': concept_arc.move_test_to_train().repeat(128),
            }
            print("Done.")


            # Epoch Limit = 142 (at the moment)
            # original_n = 644 #epochs
            print("-"*20)
            print("Loading ReArc...")
            epochs = 141
            train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=epochs, sizes=[6], seed=42, mix_datasets=mix_datasets)
            print("Done.")

            # augment data set and transform to list (eventually removing examples to stay below the max. token count)
            print("-"*20)
            print("Augmenting Data...")
            do_aug=True
            train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)

            if args.augmentation_level=="none":
                print("Not Augmenting!")
                do_aug = False

            if args.augmentation_level=="tp":
                train_aug_opts = dict(tp=True, rt=False, perm=False, shfl_ex=False, seed=0)
                print(f"Augmentation Options: {train_aug_opts}")

            if args.augmentation_level=="rt":
                train_aug_opts = dict(tp=False, rt=True, perm=False, shfl_ex=False, seed=0)
                print(f"Augmentation Options: {train_aug_opts}")

            if args.augmentation_level=="perm":
                train_aug_opts = dict(tp=False, rt=False, perm=True, shfl_ex=False, seed=0)
                print(f"Augmentation Options: {train_aug_opts}")

            if args.augmentation_level=="shfl":
                train_aug_opts = dict(tp=False, rt=False, perm=False, shfl_ex=True, seed=0)
                print(f"Augmentation Options: {train_aug_opts}")

            if args.augmentation_level=="all":
                train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
                print(f"Augmentation Options: {train_aug_opts}")

            if do_aug:
                train_dataset_augment = train_dataset.augment(**train_aug_opts)
                train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)
            print("Done.")

            # run training
            print("-"*20)
            print("Training...")
            FastLanguageModel.for_training(model)
            tokenizer.padding_side = 'right'
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=Dataset.from_list(train_dataset_as_list),
                dataset_text_field="text",
                max_seq_length=fmt_opts['max_tokens'],
                packing=False,
                data_collator=InputMaskingDataCollator(
                    instruction_template=fmt_opts['query_beg'],
                    response_template=fmt_opts['reply_beg'],
                    mlm=False,
                    tokenizer=tokenizer,
                    mask_first_n_examples=1,
                ),
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=2,
                    warmup_ratio=0.25,
                    num_train_epochs=1,
                    learning_rate=1e-4,
                    embedding_learning_rate=1e-5,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=10,
                    optim="adamw_8bit",
                    weight_decay=0.00,
                    lr_scheduler_type='cosine',
                    seed=42,
                    output_dir='tmp_output',
                    save_strategy='no',
                    report_to='none',
                ),
            )
            trainer_stats = unsloth_train(trainer)
            print("Done.")

            print("-"*20)
            print("Saving Model & Tokenizer...")
            save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)
            print("Done.")

        if action == 'merge':
            print("="*40)
            print("MERGE LOOP")
            # load peft weights and merge
            print("-"*20)
            print("Loading model...")
            load_peft_state(model, f'{save_model_path}-lora')
            print("Done.")

            print("-"*20)
            print("Merging model...")
            model = merge_peft_into_base(model)
            print("Done.")

            print("-"*20)
            print("Merging model...")
            save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)
            print("Saving.")