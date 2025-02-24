import os
import json
import datetime
# from common_stuff import (
#     RemapCudaOOM, 
#     MyFormatter, 
#     Decoder, 
#     use_aug_score, 
#     arc_test_set, 
#     score_temp_storage, 
#     aug_score_params, 
#     submission_select_algo,
#     selection_algorithms,
#     start_training,
#     start_inference,
#     infer_temp_storage,
# )

from common_stuff import (
    RemapCudaOOM, 
    Trainer, 
    MyFormatter, 
    Decoder,
    arc_test_set
)

NUM_GPUS=1
os.environ["WANDB_DISABLED"] = "true"

def produce_answers(submission_filename=None):
    """Function to produce answers"""
    if not submission_filename:
        submission_filename = "submission.json"

    with RemapCudaOOM():
        model, formatter, dataset = None, MyFormatter(), None
        decoder = Decoder(formatter, arc_test_set.split_multi_replies(), n_guesses=2, frac_score=True).from_store(infer_params['store'])
        if use_aug_score or arc_test_set.is_fake: decoder.calc_augmented_scores(model=model, store=score_temp_storage, **aug_score_params)
        submission = arc_test_set.get_submission(decoder.run_selection_algo(submission_select_algo))
        with open(submission_filename, 'w') as f: json.dump(submission, f)
        if arc_test_set.is_fake:
            decoder.benchmark_selection_algos(selection_algorithms)
            with open(submission_filename) as f: reload_submission = json.load(f)
            print('*** Reload score:', arc_test_set.validate_submission(reload_submission))




if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--augment_train", action="store_true", help="Whether to augment during training")
    parser.add_argument("--augment_evaluate", action="store_true", help="Whether to augment during inferece")
    parser.add_argument("--turbo_inference", action="store_true", help="Whether to use turbo inference")
    parser.add_argument("--min_prob", type=float, help="What min prob should be. (e.g. 0.17)")

    # parser.add_argument("--submission_filename", type=str, help="Submission Filename")
    parser.add_argument("--experiment_name", type=str, help="Submission Filename")

    args = parser.parse_args()
    # TODO:
    # infer_params = dict(min_prob=0.17, store=infer_temp_storage, use_turbo=True)
    # infer_params = dict(store=infer_temp_storage, use_turbo=True)

    if args.min_prob or args.turbo_inference:
        turbo_str = "_turbo_"
        infer_params = dict(min_prob=0.17, use_turbo=True)
    else:
        turbo_str = ""
        infer_params = dict(use_turbo=True)
    
    train_aug = "_train_aug_" if args.augment_train else ""
    eval_aug = "_eval_aug_" if args.augment_evaluate else ""

    current_time_string = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")    
    name = args.experiment_name + train_aug + eval_aug + turbo_str
    full_name = name+"_"+current_time_string
    submission_filename = name +  f"_submission_{current_time_string}.json"

    trainer = Trainer(full_name)
    trainer.start_training(gpu=0, augment=args.augment_train)
    trainer.start_inference(gpu=0, augment=args.augment_evaluate, infer_params = infer_params)

    produce_answers(submission_filename)