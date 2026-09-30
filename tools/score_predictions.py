#!/usr/bin/env python3
"""Score COCO bounding-box predictions over the entire annotated image set."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys


METRIC_NAMES = (
    'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large',
)


def read_json(path: Path):
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()


def unique_ids(records, field):
    if not isinstance(records, list) or not records:
        raise ValueError(f'annotations.{field} must be a non-empty list')
    identifiers = []
    for record in records:
        if not isinstance(record, dict) or type(record.get('id')) is not int:
            raise ValueError(f'annotations.{field} must contain integer IDs')
        identifiers.append(record['id'])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f'annotations.{field} contains duplicate IDs')
    return set(identifiers)


def validate_predictions(predictions, image_ids, category_ids):
    if not isinstance(predictions, list):
        raise ValueError('predictions must be a COCO detection JSON array')
    for index, detection in enumerate(predictions):
        prefix = f'prediction[{index}]'
        if not isinstance(detection, dict):
            raise ValueError(f'{prefix} must be an object')
        for field, valid_ids in (('image_id', image_ids), ('category_id', category_ids)):
            identifier = detection.get(field)
            if type(identifier) is not int or identifier not in valid_ids:
                raise ValueError(f'{prefix} has unknown or invalid {field}: {identifier!r}')
        bbox = detection.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f'{prefix}.bbox must be [x, y, width, height]')
        values = bbox + [detection.get('score')]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError(f'{prefix} must have finite numeric bbox coordinates and score')
        if bbox[2] < 0 or bbox[3] < 0:
            raise ValueError(f'{prefix} has negative box width or height')


def evaluate_predictions(annotations_path: Path, predictions_path: Path) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    annotations, annotation_hash = read_json(annotations_path)
    predictions, prediction_hash = read_json(predictions_path)
    if not isinstance(annotations, dict):
        raise ValueError('annotations must be a COCO ground-truth JSON object')
    image_ids = unique_ids(annotations.get('images'), 'images')
    category_ids = unique_ids(annotations.get('categories'), 'categories')
    ground_truth = annotations.get('annotations')
    if not isinstance(ground_truth, list):
        raise ValueError('annotations.annotations must be a list')
    if ground_truth:
        unique_ids(ground_truth, 'annotations')
    for annotation in ground_truth:
        if annotation.get('image_id') not in image_ids or annotation.get('category_id') not in category_ids:
            raise ValueError('ground-truth annotation references an unknown image/category')
    validate_predictions(predictions, image_ids, category_ids)

    # COCO diagnostics go to stderr so stdout remains machine-readable JSON.
    with contextlib.redirect_stdout(sys.stderr):
        coco_gt = COCO()
        coco_gt.dataset = annotations
        coco_gt.dataset.setdefault('info', {})
        coco_gt.createIndex()
        if predictions:
            coco_dt = coco_gt.loadRes(predictions)
        else:
            # COCO.loadRes([]) indexes the first detection; build an empty set directly.
            coco_dt = COCO()
            coco_dt.dataset = {
                'images': annotations['images'],
                'categories': annotations['categories'],
                'annotations': [],
            }
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        # Include images with no predictions; do not infer the split from detections.
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = sorted(category_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        'metrics': dict(zip(METRIC_NAMES, map(float, evaluator.stats))),
        'evaluation': {
            'iou_type': 'bbox',
            'metric_scale': '0-1; -1 means unavailable',
            'image_count': len(image_ids),
            'ground_truth_count': len(ground_truth),
            'prediction_count': len(predictions),
            'category_ids': sorted(category_ids),
            'iou_thresholds': evaluator.params.iouThrs.round(2).tolist(),
            'max_detections': list(evaluator.params.maxDets),
            'use_categories': bool(evaluator.params.useCats),
        },
        'inputs': {
            'annotations': {'file': annotations_path.name, 'sha256': annotation_hash},
            'predictions': {'file': predictions_path.name, 'sha256': prediction_hash},
        },
        'environment': {
            'python': platform.python_version(),
            'numpy': version('numpy'),
            'pycocotools': version('pycocotools'),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='AP/AR use the 0-1 scale (multiply by 100 for percent); -1 denotes an unavailable metric.',
    )
    parser.add_argument('--annotations', required=True, type=Path, help='COCO ground-truth JSON for the full evaluation split')
    parser.add_argument('--predictions', required=True, type=Path, help='COCO detections: image_id, category_id, bbox (xywh), score')
    parser.add_argument('--output', type=Path, help='also save metrics, input hashes, and environment as JSON')
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in {args.annotations.resolve(), args.predictions.resolve()}:
            raise ValueError('--output must not overwrite either input file')
        report = evaluate_predictions(args.annotations, args.predictions)
        serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8', newline='\n')
        print(serialized, end='')
    except ImportError as error:
        parser.error(f'{error}. Install dependencies with: python -m pip install -r requirements-eval.txt')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
