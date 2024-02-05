import json
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict

import faiss
import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

import copy
import time
from combiner_train import extract_index_features
from data_utils import CIRRDataset, targetpad_transform, squarepad_transform, base_path
from combiner import Combiner
from utils import element_wise_sum, device


def generate_cirr_test_submissions(combining_function: callable, file_name: str, clip_model: CLIP,
                                   preprocess: callable, faisstype):
    """
   Generate and save CIRR test submission files to be submitted to evaluation server
   :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
   :param file_name: file_name of the submission
   :param clip_model: CLIP model
   :param preprocess: preprocess pipeline
   """

    clip_model = clip_model.float().eval()

    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset('test1', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_test_dataset, clip_model)
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess)
                                     
    name_to_feat = dict(zip(index_names, index_features))
    if faisstype == 'PCA+IVFHNSW':
        faiss_index = faiss.index_factory(640, "PCA320,IVF10_HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
        faiss1 = faiss.index_factory(640, "PCA320,IVF10_HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
    elif faisstype == 'PCA':
        faiss_index = faiss.index_factory(640, "PCA320,Flat", faiss.METRIC_INNER_PRODUCT)
        faiss1 = faiss.index_factory(640, "PCA320,Flat", faiss.METRIC_INNER_PRODUCT)
    elif faisstype == 'IVFHNSW':
        faiss_index = faiss.index_factory(640, "IVF10_HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
        faiss1 = faiss.index_factory(640, "IVF10_HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
    else:
        faiss_index = faiss.index_factory(640, "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss1 = faiss.index_factory(640, "Flat", faiss.METRIC_INNER_PRODUCT)
    st = time.time()
    if faisstype in ['PCA+IVFHNSW', 'PCA', 'IVFHNSW']:
        faiss1.train(F.normalize(torch.tile(index_features, (100, 1)), dim=-1).float().cpu().detach().numpy())
    ed = time.time()
    faiss_index.train(F.normalize(index_features, dim=-1).float().cpu().detach().numpy())
    print('Train time: ', ed - st)
    faiss_index.add(F.normalize(index_features, dim=-1).float().cpu().detach().numpy())
    

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset, clip_model,
                                                                                  faiss_index, faiss1, index_names,
                                                                                  combining_function, name_to_feat)
    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = base_path / "submission" / 'CIRR'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)

def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset, clip_model: CLIP, faiss_index, faiss1,
                             index_names: List[str], combining_function: callable, name_to_feat) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: test index features
    :param index_names: test index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id = \
        generate_cirr_test_predictions(clip_model, relative_test_dataset, combining_function, index_names,
                                       faiss_index, name_to_feat)

    print(f"Compute CIRR prediction dicts")

    # Compute the distances and sort the results
    st = time.time()
    score, sorted_indices = faiss1.search(predicted_features.cpu().detach().numpy(), 60)
    ed = time.time()
    score, sorted_indices = faiss_index.search(predicted_features.cpu().detach().numpy(), 60)
    print('Search time: ', ed - st)

    new_sorted_indices = []
    for samp in sorted_indices:
        new_sorted_indices.append(np.concatenate((np.array(samp), np.array(list(set(range(len(index_names))) - set(samp))))))
    sorted_indices = np.array(new_sorted_indices)
    sorted_index_names = np.array(index_names)[sorted_indices]
    
    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    

    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(clip_model: CLIP, relative_test_dataset: CIRRDataset, combining_function: callable,
                                   index_names: List[str], faiss_index, name_to_feat) -> \
        Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    """
    Compute CIRR predictions on the test set
    :param clip_model: CLIP model
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
    :param index_features: test index features
    :param index_names: test index names

    :return: predicted_features, reference_names, group_members and pairs_id
    """
    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=True)


    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(
            relative_test_loader):  # Load data
        text_inputs = clip.tokenize(captions, context_length=77).to(device)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    return predicted_features, reference_names, group_members, pairs_id


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="submission file name")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=str, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--faiss", default="None", type=str,
                        help="Preprocess pipeline, should be in ['None', 'PCA', 'IVFHSNW', 'PCA+IVFHSNW'] ")
    args = parser.parse_args()
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess
        
    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device)
        saved_state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(saved_state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    generate_cirr_test_submissions(combining_function, args.submission_name, clip_model, preprocess, args.faiss)


if __name__ == '__main__':
    main()
