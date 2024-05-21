"""Postprocessing for R-CNN prediction."""

from typing import Dict

import torch
from torchvision.ops.boxes import box_area, _box_inter_union
from skimage.measure import regionprops


def finetune_bboxes(masks: torch.Tensor, threshold: float = .5) -> torch.Tensor:
    """
    Recalculates bounding boxes from the masks.

    Args:
        masks: tensor with all the predicted masks
        threshold: threshold for mask binarisation

    Returns:
        new bounding boxes
    """
    np_masks = (masks > threshold).to(int).numpy()  # type: ignore
    new_bboxes = torch.zeros((len(np_masks), 4))
    for i, mask in enumerate(np_masks):
        new_bboxes[i] = torch.tensor(list(regionprops(np_masks[0])[0].bbox))
        new_bboxes[i] = new_bboxes[i][torch.tensor([1, 0, 3, 2])]
    return new_bboxes


def postprocess(prediction: Dict[str, torch.Tensor],
                threshold: float = 0.5,
                method: str = 'iou') -> Dict[str, torch.Tensor]:
    """
    Postprocesses the predicted boxes using a non maxima suppression for overlapping boxes.

    Args:
        prediction: output of the model
        threshold: threshold for overlap
        method: method to reduce bounding boxes can be non maxima suppression with 'iou'
                and Intersection over minmum ('iom')

    Raises:
        NameError: if given method is not 'iou' or 'iom'.

    Returns:
        Dictionary with reduced predictions
    """
    if method not in ('iou', 'iom'):
        raise NameError(f"Method must be one of 'iou' or 'iom', got {method}!")

    scores = prediction['scores']
    boxes = finetune_bboxes(prediction['masks'])

    # calc matrix of intersection depending on method
    inter, union = _box_inter_union(boxes, boxes)
    area = box_area(prediction['boxes'])
    min_matrix = torch.min(area.unsqueeze(1), area.unsqueeze(0))
    matrix: torch.Tensor = inter / min_matrix if method == 'iom' else inter / union
    matrix.fill_diagonal_(0)

    # indices of intersections over threshold
    indices = (matrix > threshold).nonzero()

    # calc box indices to keep
    values = scores[indices]
    drop_indices = indices[torch.arange(len(indices)), torch.argmin(values, dim=1)]
    keep_indices = torch.tensor(list(set(range(len(scores))) - set(drop_indices.tolist())))
    keep_indices = keep_indices.to(int)     # type: ignore
    # remove non maxima
    reduced_prediction = {
        'boxes': boxes[keep_indices],
        'scores': scores[keep_indices],
        'masks': prediction['masks'][keep_indices]
    }

    # make boxes int and mask bool
    reduced_prediction["boxes"] = reduced_prediction["boxes"].int()
    reduced_prediction["masks"] = reduced_prediction["masks"] > 0.5

    # sort results
    order = [index for index, _ in sorted(enumerate(reduced_prediction['boxes']),
                                          key=lambda box: (box[0], box[1]))]
    reduced_prediction['boxes'] = reduced_prediction['boxes'][order]
    reduced_prediction['scores'] = reduced_prediction['scores'][order]
    reduced_prediction['masks'] = reduced_prediction['masks'][order]

    return reduced_prediction


if __name__ == '__main__':
    import torch

    # Example vector and indices (as PyTorch tensors)
    vector = torch.tensor([5, 3, 8, 2, 7])
    indices = torch.tensor([[0, 1], [2, 4], [3, 0]])

    # Extract values from vector using indices
    values = vector[indices]

    # Find the index of the minimum value along the second axis (axis=1)
    min_index = torch.argmin(values, dim=1)

    # Get the original indices
    original_indices = indices[torch.arange(len(indices)), min_index]

    # Get positions not in original_indices
    not_in_original_indices = torch.tensor(list(set(range(len(vector))) -
                                                set(original_indices.tolist())))

    print("Positions of the vector not in original_indices:", not_in_original_indices)
