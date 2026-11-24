from pathlib import Path

import cv2
import numpy as np
import torch

from GSAFD import AFD_NET
from args import Args as args
import utils


# Run from the repository root and edit these paths for your data.
test_path = './test/ir/'
model_path = './model/afd_net.model'
output_path_root = './outputs/M3FD/'


def load_model(path, num_block, fusion_type='cat'):
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Set Args.cuda = 0 for CPU inference.')
    device = torch.device('cuda' if args.cuda else 'cpu')
    model = AFD_NET(args.s, args.n, args.channel, args.stride, num_block, fusion_type)
    print(f'Loading model from: {path}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def run(model, infrared_path, visible_path, output_path, img_flag, img_name):
    infrared = utils.get_image(str(infrared_path), height=None, width=None, flag=False)
    visible_bgr = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
    if visible_bgr is None:
        raise FileNotFoundError(f'Cannot read visible image: {visible_path}')
    if infrared.shape != visible_bgr.shape[:2]:
        raise ValueError(f'IR and visible dimensions differ: {infrared_path}, {visible_path}')

    visible_ycrcb = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2YCrCb)
    device = torch.device('cuda' if args.cuda else 'cpu')
    infrared_tensor = torch.from_numpy(infrared).float()[None, None].to(device) / 255.0
    visible_tensor = torch.from_numpy(visible_ycrcb[:, :, 0]).float()[None, None].to(device) / 255.0

    fused = model(infrared_tensor, visible_tensor)['fuse']
    if fused.shape != infrared_tensor.shape:
        raise ValueError(f'Unexpected fused image shape for {infrared_path}: {tuple(fused.shape)}')
    fused_y = fused[0, 0].detach().cpu().numpy()
    if not np.isfinite(fused_y).all():
        raise ValueError(f'Nonfinite fused pixels for {infrared_path}')
    fused_y = (np.clip(fused_y, 0.0, 1.0) * 255.0).astype(np.uint8)
    fused_ycrcb = cv2.merge([fused_y, visible_ycrcb[:, :, 1], visible_ycrcb[:, :, 2]])
    fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)

    destination = Path(output_path) / f'{img_name}.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), fused_bgr):
        raise OSError(f'Could not save fused image: {destination}')
    print(f'Saved {destination}')
    return destination


def main():
    infrared_dir = Path(test_path).resolve()
    visible_dir = infrared_dir.parent / 'vis'
    output_dir = Path(output_path_root).resolve()
    if not infrared_dir.is_dir() or not visible_dir.is_dir():
        raise FileNotFoundError(f'Expected paired input folders: {infrared_dir}, {visible_dir}')
    if not Path(model_path).is_file():
        raise FileNotFoundError(f'Model file not found: {model_path}')
    if output_dir.is_relative_to(infrared_dir) or output_dir.is_relative_to(visible_dir):
        raise ValueError('Choose an output directory outside both input folders.')

    image_paths, _ = utils.list_images(str(infrared_dir))
    image_paths = sorted(Path(path) for path in image_paths)
    if not image_paths:
        raise ValueError(f'No input images found in {infrared_dir}')

    output_names = set()
    for infrared_path in image_paths:
        relative_path = infrared_path.relative_to(infrared_dir)
        visible_path = visible_dir / relative_path
        if not visible_path.is_file():
            raise FileNotFoundError(f'Matching visible image not found: {visible_path}')
        output_name = relative_path.with_suffix('.png').as_posix().casefold()
        if output_name in output_names:
            raise ValueError(f'Multiple inputs would overwrite the same output: {relative_path}')
        output_names.add(output_name)

    model = load_model(model_path, args.num_block)
    for infrared_path in image_paths:
        relative_path = infrared_path.relative_to(infrared_dir)
        run(model, infrared_path, visible_dir / relative_path,
            output_dir / relative_path.parent, args.channel != 1, relative_path.stem)
    print(f'Finished. Saved {len(image_paths)} fused images in {output_dir}')


if __name__ == '__main__':
    main()
