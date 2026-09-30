"""Prepare paired images and COCO annotations without modifying source data."""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
M3FD_SOURCES = {
    'train': '1TZWiVsPaAuL_yxDEDbuLMdY6fa6JXYnn',
    'val': '1emERAwm1AIV24Uc6t8OnhZDeNt3fjJy4',
}


def first_existing(paths, label):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f'{label} not found; checked: ' + ', '.join(map(str, paths)))


def safe_name(name):
    p = PurePosixPath(name.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or ':' in str(p):
        raise ValueError(f'annotation filename must be relative: {name!r}')
    return Path(*p.parts)


def paired_paths(name, vis, ir, dataset=None):
    vis_path, ir_path = vis/name, ir/name
    if dataset == 'flir':
        if not vis_path.is_file():
            vis_path = vis/name.with_name(name.stem + '_RGB.jpg')
        if not ir_path.is_file():
            ir_path = ir/name.with_name(name.stem + '_PreviewData.jpeg')
    return vis_path, ir_path


def validate_coco(coco, vis, ir, dataset=None):
    images = coco['images']
    ids = [int(x['id']) for x in images]
    names = [x['file_name'] for x in images]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError('duplicate image IDs or filenames')
    cats = [int(x['id']) for x in coco['categories']]
    if len(cats) != len(set(cats)):
        raise ValueError('duplicate category IDs')
    image_ids, cat_ids = set(ids), set(cats)
    for ann in coco['annotations']:
        if ann['image_id'] not in image_ids or ann['category_id'] not in cat_ids:
            raise ValueError('annotation references an unknown image/category')
        if len(ann['bbox']) != 4:
            raise ValueError('expected COCO xywh boxes')
    for image in images:
        name = safe_name(image['file_name'])
        vis_path, ir_path = paired_paths(name, vis, ir, dataset)
        if not vis_path.is_file() or not ir_path.is_file():
            raise FileNotFoundError(f'missing VIS/IR pair: {name}')
    return images


def transfer(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'symlink':
        dst.symlink_to(src.resolve())
    else:
        shutil.copyfile(src, dst)


def coco_paths(source, dataset, split, annotations_dir=None, vis_override=None, ir_override=None):
    source_split = 'test' if dataset == 'llvip' and split == 'val' else split
    ann_root = annotations_dir or source/'annotations'
    filenames = [f'{split}.json', f'{source_split}.json', f'instances_{split}2014.json', f'm3fd_{split}_new.json']
    annotation = next((ann_root/n for n in filenames if (ann_root/n).is_file()), None)
    raw_dirs = [source/'JPEGImages'] if dataset == 'flir' else []
    vis = first_existing([vis_override] if vis_override else [
        source/f'{split}_imgs'/'vis_imgs', source/f'{split}_vis_img',
        source/'visible'/source_split, source/'vis'/source_split,
        source/'Vis', source/'VIS', source/'vis', source/'VI'] + raw_dirs, 'VIS images')
    ir = first_existing([ir_override] if ir_override else [
        source/f'{split}_imgs'/'ir_imgs', source/f'{split}_ir_img',
        source/'infrared'/source_split, source/'ir'/source_split,
        source/'Ir', source/'IR', source/'ir'] + raw_dirs, 'IR images')
    return annotation, vis, ir


def prepare(dataset, source, output, annotations_dir=None, mode='copy', vis_dir=None, ir_dir=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('output must be separate from the original dataset')
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    annotations_dir = Path(annotations_dir).resolve() if annotations_dir else None
    plans, report = [], {'dataset':dataset,'splits':{}}
    for split in ['train','val']:
        ann,vis,ir=coco_paths(source,dataset,split,annotations_dir,
                             Path(vis_dir).resolve() if vis_dir else None,
                             Path(ir_dir).resolve() if ir_dir else None)
        if ann:
            raw=ann.read_bytes()
            coco=json.loads(raw.decode('utf-8-sig'))
            annotation_mode='provided_coco'
        elif dataset=='m3fd':
            file_id=M3FD_SOURCES[split]
            with urllib.request.urlopen('https://drive.google.com/uc?export=download&id='+file_id,timeout=90) as response:
                raw=response.read()
            coco=json.loads(raw)
            annotation_mode='eaef_coco'
        else:
            raise FileNotFoundError(f'{dataset.upper()} requires COCO annotations; pass --annotations-dir. See README.md')
        if len(coco['categories']) != {'flir':3,'m3fd':6,'llvip':1}[dataset]:
            raise ValueError('category count differs from the selected training configuration')
        images=validate_coco(coco,vis,ir,dataset)
        if not images:
            raise ValueError(f'empty {split} split')
        names={x['file_name'] for x in images}
        if split=='train': train_names=names
        elif dataset in {'m3fd','llvip'} and train_names & names:
            raise ValueError('train/evaluation filename overlap')
        if dataset=='m3fd':
            target_vis,target_ir=f'{split}_vis_img',f'{split}_ir_img'
            target_ann=f'annotations/m3fd_{split}_new.json'
        else:
            target_vis,target_ir=f'{split}_imgs/vis_imgs',f'{split}_imgs/ir_imgs'
            target_ann=f'annotations/{split}.json'
        plans.append((raw,images,vis,ir,target_vis,target_ir,target_ann))
        report['splits'][split]={'images':len(images),'annotations':len(coco['annotations']),
            'categories':coco['categories'],'annotation_mode':annotation_mode}
    if report['splits']['train']['categories']!=report['splits']['val']['categories']:
        raise ValueError('train and evaluation category ordering differs')
    # All source pairs and metadata are checked before creating the output.
    output.mkdir(parents=True)
    for raw,images,vis,ir,tvis,tir,tann in plans:
        for image in images:
            name=safe_name(image['file_name'])
            vis_path,ir_path=paired_paths(name,vis,ir,dataset)
            transfer(vis_path,output/tvis/name,mode)
            transfer(ir_path,output/tir/name,mode)
        (output/tann).parent.mkdir(parents=True,exist_ok=True)
        (output/tann).write_bytes(raw)
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['flir','m3fd','llvip'])
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--annotations-dir',type=Path)
    parser.add_argument('--vis-dir',type=Path,help='flat VIS image directory, when not using a standard layout')
    parser.add_argument('--ir-dir',type=Path,help='flat IR image directory, when not using a standard layout')
    parser.add_argument('--mode',choices=['copy','symlink'],default='copy')
    args=parser.parse_args()
    print(json.dumps(prepare(args.dataset,args.source,args.output,args.annotations_dir,args.mode,args.vis_dir,args.ir_dir),indent=2))


if __name__=='__main__':
    main()
