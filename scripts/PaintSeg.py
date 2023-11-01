import argparse, os, sys, glob
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2
import mmcv
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from kmeans_pytorch import kmeans
from sklearn.decomposition import PCA
from torchvision import transforms
sys.path.append('./mae')
import models_mae
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
from scipy import ndimage
import random
from mask2scribble import *
import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline

seed = 123
root = 'PaintSeg'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def densecrf(image, mask):
    MAX_ITER = 10
    POS_W = 7
    POS_XY_STD = 3
    Bi_W = 10
    Bi_XY_STD = 50
    Bi_RGB_STD = 5
    mask = mask.cpu().numpy()
    h, w = mask.shape
    mask = mask.reshape(1, h, w)
    fg = mask.astype(np.float)
    bg = 1 - fg
    output_logits = torch.from_numpy(np.concatenate((bg, fg), axis=0))

    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear").squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    MAP = np.argmax(Q, axis=0).reshape((h, w)).astype(np.float32)
    return MAP

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print('loaded ckpts for vit mae', msg)
    return model

def make_batch(image, mask, device):
    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

def rescale(x):
    # rescale normalized prediction to image
    return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

def toPIL(x):
    # x = rescale(x)
    x = x.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    x = Image.fromarray(x.astype(np.uint8)) #.convert('RGB')
    # x.save('debug.png')
    return x

def inpaint_masked_image(image, mask, model, device, return_img_mask=False):
    batch = make_batch(image, mask, device=device)
    # encode masked image and concat downsampled mask
    c = model.cond_stage_model.encode(batch["masked_image"])
    cc = torch.nn.functional.interpolate(batch["mask"],
                                         size=c.shape[-2:])
    c = torch.cat((c, cc), dim=1)

    shape = (c.shape[1] - 1,) + c.shape[2:]
    samples_ddim, _ = sampler.sample(S=opt.steps,
                                     conditioning=c,
                                     batch_size=c.shape[0],
                                     shape=shape,
                                     verbose=False)
    predicted_image = model.decode_first_stage(samples_ddim)
    predicted_image = rescale(predicted_image)
    image = batch["image"]
    mask = batch["mask"]
    inpainted = (1 - mask) * image + mask * predicted_image
    # inpainted = inpainted

    # if return_img_mask:
    #     image = rescale(batch["image"]) # .cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    #     mask = rescale(batch["mask"]) # .cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    #     return inpainted, image, mask
    return inpainted

def visualize(x, path):
    x = x.clamp(0, 1)
    x = x.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    if x.shape[-1] == 1:
        x = x[:, :, 0]
    Image.fromarray(x.astype(np.uint8)).save(path)

class EMA:
    def __init__(self, beta, mask):
        self.value = None
        self.beta = beta
        self.mask = mask
    def update(self, x, pos):
        if self.value is None:
            self.value = torch.zeros_like(self.mask)
            self.value[:, :, pos[0]:pos[1], pos[2]:pos[3]] = x
        else:
            self.value[:, :, pos[0]:pos[1], pos[2]:pos[3]] = \
                self.beta * self.value[:, :, pos[0]:pos[1], pos[2]:pos[3]] + (1 - self.beta) * x
    def get_value(self):
        value = self.value.clone()
        return value

def get_negihbour(mask, device, iteration=4, kernel_size=3):
    mask = mask.cpu().numpy().astype(np.uint8)[0, 0]
    dilate = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)), iterations=iteration)[None, None, ...]
    erode = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)), iterations=iteration)[None, None, ...]
    dilate = torch.from_numpy(dilate).to(device)
    erode = torch.from_numpy(erode).to(device)
    neighbour = dilate - erode
    return neighbour, dilate, erode

def get_dilate(mask, device, iteration=4, kernel_size=3):
    mask = mask.cpu().numpy().astype(np.uint8)[0, 0]
    dilate = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)), iterations=iteration)[None, None, ...]
    dilate = torch.from_numpy(dilate).to(device)
    return dilate

def get_erode(mask, device, iteration=4, kernel_size=3):
    mask = mask.cpu().numpy().astype(np.uint8)[0, 0]
    erode = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)), iterations=iteration)[None, None, ...]
    erode = torch.from_numpy(erode).to(device)
    return erode

def feature_pyramid(image, backbone, scales=((1, 1),)): # , (2, 1), (1, 2),(2, 2)
    _, _, h, w = image.shape
    features = []
    # scale, (h_scale, w_scale)
    for scale in scales:
        image_hw = [image_h.chunk(scale[1], dim=3) for image_h in image.chunk(scale[0], dim=2)]
        image_hw_ = []
        for image_h in image_hw:
            image_h_ = []
            for image_ in image_h:
                image_feat = backbone.get_intermediate_layers(dinoTransform(toPIL(image_))[None, ...].to(device), 12)[-1]
                if opt.dino_v2:
                    image_feat = image_feat.permute((0, 2, 1)).reshape(1, 768, 60, 60)
                else:
                    image_feat = image_feat[:, 1:, :].permute((0, 2, 1)).reshape(1, 768, 60, 60)
                image_feat = torch.nn.functional.interpolate(image_feat, (500//scale[0], 500//scale[1]))
                image_h_.append(image_feat)
            image_h_ = torch.cat(image_h_, dim=3)
            image_hw_.append(image_h_)
        image_hw_ = torch.cat(image_hw_, dim=2)
        features.append(image_hw_)
    features = torch.cat(features, dim=1)
    return features

def predict_mask(iter, backbone, inpainter_fg, inpainter_bg, image, mask, dinoTransform, save_path,
                 add_perturbation=False, use_crf=False, global_cluster=False, points=None):
    # if even iter (from 0) mask foreground else mask background
    # add perturbation to the mask boundary
    ori_mask = mask
    ema = EMA(beta=0.8, mask=mask)
    with_box_constraint = True
    offset_range = 4
    index = torch.where(mask == 1)
    a, aa, b, bb = index[2].min(), index[2].max(), index[3].min(), index[3].max()
    ca, cb = (a + aa) / 2, (b + bb) / 2
    da, db = (aa - a) / 2, (bb - b) / 2
    # constraint 1.2 times box as valid area
    valid_a, valid_aa, valid_b, valid_bb = int(ca - 1.25 * da), int(ca + 1.25 * da), int(cb - 1.25 * db), int(cb + 1.25 * db)
    valid_a, valid_aa, valid_b, valid_bb = max(0, valid_a), min(512, valid_aa), max(0, valid_b), min(512, valid_bb)
    valid_mask = torch.zeros_like(mask)
    valid_mask[:, :, valid_a:valid_aa, valid_b:valid_bb] = 1

    for i in range(iter):
        print(i)
        # find bbox
        flag = 1 if i % 2 == 0 else 0
        if flag == 1 and not opt.gt:  # if foreground stage, update box
            index = torch.where(mask == flag)
            a, aa, b, bb = index[2].min(), index[2].max(), index[3].min(), index[3].max()  # TODO: 空mask会挂掉
        # multiple scale for clustering
        factors = [1.1, 1.1, 1.1, 1.1, 1.1]
        ca, cb = (a + aa) / 2, (b + bb) / 2
        da, db = (aa - a) / 2, (bb - b) / 2
        boxes, crop_boxes, diffs = [], [], []

        # for background inpainting
        if i % 2 != 0:
            if opt.point_num == 0:
                dilate = get_dilate(1 - mask, device, iteration=1, kernel_size=15)
                dilate = dilate.float()
                inpaint_mask = (mask * dilate).cpu().numpy()[0, 0] * 255
            else:
                inpaint_mask = (mask).cpu().numpy()[0, 0] * 255
            inpaint_mask = Image.fromarray(inpaint_mask.astype(np.uint8))
            inpainted_images = inpainter_bg(
                prompt='',
                image=toPIL(image),
                mask_image=inpaint_mask,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                num_images_per_prompt=len(factors),
            ).images

        for factor_id, factor in enumerate(factors):
            a, aa, b, bb = int(ca - factor * da), int(ca + factor * da), int(cb - factor * db), int(cb + factor * db)
            boxes.append((a, aa, b, bb))

            # dilate fg mask when inpaint fg (bg inpainting always lead to eroded fg mask)
            if i % 2 == 0:
                dilate = get_dilate(mask, device, iteration=4)
                inpaint_mask = dilate.float()
                ori_inpained = inpaint_masked_image(image, inpaint_mask, inpainter_fg, device)
                inpainted = inpaint_mask * ori_inpained + (1 - inpaint_mask) * image
            else:
                inpainted = inpainted_images[factor_id]
                inpainted = np.array(inpainted).astype(np.float32) / 255.0
                inpainted = inpainted[None].transpose(0, 3, 1, 2)
                inpainted = torch.from_numpy(inpainted).to(device)

            visualize(inpainted, os.path.join(save_path, f'{i}_{factor_id}.jpg'))

            offset = np.random.randint(-offset_range, offset_range)
            if with_box_constraint:
                crop_a, crop_aa, crop_b, crop_bb = max(0, a+offset), min(512, aa+offset), max(0, b+offset), min(512, bb+offset)
            else:
                crop_a, crop_aa, crop_b, crop_bb = max(0, valid_a+offset), min(512, valid_aa+offset), max(0, valid_b+offset), min(512, valid_bb+offset)
                # 0, 512, 0, 512
            crop_boxes.append((crop_a, crop_aa, crop_b, crop_bb))

            inpainted_ = inpainted[:, :, crop_a:crop_aa, crop_b:crop_bb]
            image_ = image[:, :, crop_a:crop_aa, crop_b:crop_bb]
            mask_ = mask[:, :, crop_a:crop_aa, crop_b:crop_bb]
            # cal diff
            # [b, 3601, 768]
            inpainted_feat = feature_pyramid(inpainted_, backbone)
            image_feat = feature_pyramid(image_, backbone)
            # cal diff
            diff = torch.sum((inpainted_feat - image_feat) ** 2, dim=1, keepdim=True)  # [b, c, h, w]

            if use_crf:
                ori_diff = diff
                diff_min, diff_max = diff.min(), diff.max()
                image_crf = (image_[0].permute((1, 2, 0)).cpu().numpy() * 255).astype(np.uint8)
                crf_diff = densecrf(image_crf, mask_[0, 0])[None, None, ...]
                crf_diff = torch.from_numpy(crf_diff).to(image.device)
                # visualize(ori_diff,  os.path.join(save_path, f'{i}_.png'))
                crf_diff = crf_diff * (diff_max - diff_min) + diff_min
                crf_diff = torch.nn.functional.interpolate(crf_diff, (500, 500))
                diff = 0.2 * crf_diff + 0.8 * ori_diff

            if opt.point_num != 0:
                mesh_x, mesh_y = torch.meshgrid(torch.linspace(0, crop_aa - crop_a, steps=crop_aa - crop_a),
                                                torch.linspace(0, crop_bb - crop_b, steps=crop_bb - crop_b))
                mesh_x, mesh_y = mesh_x.to(device), mesh_y.to(device)
                sigma = (crop_aa - crop_a + crop_bb - crop_b) / 5
                gaussian = torch.zeros_like(diff)
                for point in points:
                    mu_x, mu_y = point[0] - crop_a, point[1] - crop_b
                    gaussian_point = torch.exp(-((mesh_x - mu_x) ** 2 + (mesh_y - mu_y) ** 2) / (2 * sigma ** 2))[
                        None, None, ...]
                    gaussian_point = torch.nn.functional.interpolate(gaussian_point, (500, 500))
                    gaussian += gaussian_point
                diff_min, diff_max = diff.min(), diff.max()
                gaussian = gaussian * (diff_max - diff_min) + diff_min
                gaussian = (0.2 - i * 0.04) * gaussian
                # diff = gaussian
                diff = diff + gaussian if i % 2 == 0 else diff - gaussian
            diffs.append(diff)

            visualize(torch.nn.functional.interpolate(diff / (diff.max() - diff.min()),
                                                      (crop_aa - crop_a, crop_bb - crop_b)),
                      os.path.join(save_path, f'{i}_{factor_id}_.jpg'))

        diffs = torch.stack(diffs, dim=0)
        num_scales, batch, c, h, w = diffs.shape
        diffs = diffs.reshape((-1, 1))
        if opt.box_init:
            cluster_num = 3 if i < 1 else 2
        elif opt.point_num != 0:
            cluster_num = 3 if i < 3 else 2
        else:
            cluster_num = 2
        ids, cluster_centers = kmeans(
            X=diffs, num_clusters=cluster_num, distance='euclidean',
            device=diffs.device, iter_limit=200, tqdm_flag=False
        ) # 3 if i == 0 else 2
        new_mask_ids = ids.reshape((num_scales, batch, c, h, w))[-len(factors):]

        new_masks = []
        empty_mask = torch.zeros_like(mask)
        count_mask = torch.zeros_like(mask)
        for scale_idx, crop_box in enumerate(crop_boxes):
            crop_a, crop_aa, crop_b, crop_bb = crop_box
            crop_size = (crop_aa - crop_a, crop_bb - crop_b)
            new_mask = torch.nn.functional.interpolate(new_mask_ids[scale_idx].float(), crop_size).to(diffs.device)
            # change the new mask to the foreground mask
            if i % 2 == 0:
                # foreground masked
                cluster_id = torch.argmax(cluster_centers[:, 0])
                new_mask_ = torch.zeros_like(new_mask)
                new_mask_[new_mask == cluster_id] = 1
                new_mask = new_mask_
            else:
                # background masked
                cluster_id = torch.argmin(cluster_centers[:, 0])
                new_mask_ = torch.zeros_like(new_mask)
                new_mask_[new_mask == cluster_id] = 1
                new_mask = new_mask_
            new_masks.append(new_mask)
            empty_mask[:, :, crop_a:crop_aa, crop_b:crop_bb] += new_mask
            count_mask[:, :, crop_a:crop_aa, crop_b:crop_bb] += 1
        empty_mask[count_mask > 0] = empty_mask[count_mask > 0] / count_mask[count_mask > 0]
        # use mask obtained from this iter for the next iter
        new_mask = empty_mask  # torch.zeros_like(mask)
        visualize(new_mask, os.path.join(save_path, f'{i}.png'))

        new_mask = new_mask * valid_mask
        # FG: only reduce, BG: only increase
        if i % 2 == 0:
            dilate = get_dilate(mask, device, kernel_size=25 if opt.point_num != 0 else 5)
            new_mask = dilate * new_mask
        else:
            dilate = get_dilate(1 - mask, device, kernel_size=25 if opt.point_num != 0 else 5)
            new_mask = dilate * new_mask * mask + (1 - mask)

        new_mask_binary = torch.zeros_like(new_mask)
        new_mask_binary[new_mask > 0.5] = 1
        new_mask_binary[new_mask <= 0.5] = 0

        # filter out sparse
        erode = get_erode(new_mask_binary.float(), device, kernel_size=5, iteration=1)
        dilate = get_dilate(erode.float(), device, kernel_size=5, iteration=1)
        new_mask = new_mask * dilate.float()

        visualize(new_mask, os.path.join(save_path, f'{i}_.png'))

        # add to ema
        ema.update(new_mask, (0, 512, 0, 512))

        # binarize
        new_mask[new_mask > 0.5] = 1
        new_mask[new_mask <= 0.5] = 0

        # fill holes
        dilate = get_dilate(new_mask.float(), device, kernel_size=5, iteration=1)
        new_mask = get_erode(dilate.float(), device, kernel_size=5, iteration=1)
        new_mask = new_mask.float()

        # decide mask foreground or background in next iteration
        new_mask = 1 - new_mask if i % 2 == 0 and i != iter - 1 else new_mask

        mask = new_mask

    return mask

def pad_image(image, target_size, type='rgb'):
    """
    :param image: input image
    :param target_size: a tuple (num,num)
    :return: new image
    """
    iw, ih = image.size
    w, h = target_size

    scale = min(w / iw, h / ih)

    nw = int(iw * scale + 0.5)
    nh = int(ih * scale + 0.5)

    image = image.resize((nw, nh), Image.BICUBIC)
    if type == 'rgb':
        new_image = Image.new('RGB', target_size, (0, 0, 0))
    else:
        new_image = Image.new('L', target_size, (0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="max iteration of inpainting for segm",
    )
    parser.add_argument(
        "--use_crf",
        action='store_false',
        help="use crf in each iter",
    )
    parser.add_argument(
        "--box_init",
        action='store_true',
        help="use box as init",
    )
    parser.add_argument(
        "--scribble_init",
        action='store_true',
        help="use scribble as init",
    )
    parser.add_argument(
        "--gt",
        action='store_true',
        help="use gt mask",
    )
    parser.add_argument(
        "--point_num",
        type=int,
        default=0,
        help="point init, number of point",
    )
    parser.add_argument(
        "--dino_v2",
        action='store_true',
        help="use dino_v2",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="set split",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
    )
    opt = parser.parse_args()
    print(opt)
    setup_seed(seed)
    # image_path = 'data/inpainting_examples/photo-1583445095369-9c651e7e5d34.png'  #billow926-12-Wc-Zgx6Y.png' #6458524847_2f4c361183_k.png'  #overture-creations-5sI6fQgYIuo.png' #
    # mask_path = 'outputs/ms_box_women2/women2_mask2.png'#'data/inpainting_examples/photo-1583445095369-9c651e7e5d34_mask.png'  #6458524847_2f4c361183_k_mask.png'  #billow926-12-Wc-Zgx6Y_mask.png' #overture-creations-5sI6fQgYIuo_mask.png' #

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    model_path = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    if opt.dino_v2:
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    else:
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', force_reload=True).to(device)
    backbone.eval()
    dinoTransform = transforms.Compose([transforms.Resize((840, 840) if opt.dino_v2 else (480, 480), interpolation=3),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    image_paths = sorted(glob.glob(f'{root}/{opt.dataset}/images/*'), reverse=True)

    if opt.split != 999:
        image_paths = image_paths[opt.split::2]

    save_path = image_paths[0].replace(f'{opt.dataset}', f'PaintSeg/{opt.datasett}').replace('images', f'{opt.outdir}')
    os.makedirs(save_path, exist_ok=True)
    for image_path in tqdm(image_paths):
        (filepath, tempfilename) = os.path.split(image_path)
        (filename, extension) = os.path.splitext(tempfilename)
        # read image
        image = Image.open(image_path).convert("RGB")
        ori_shape = image.size[-2:]
        image = np.array(pad_image(image, (512, 512)))
        image = image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(device)
        # for mask_id, mask_path in enumerate(mask_paths):
        with torch.no_grad():
            # initial mask as a bounding box
            (_, tempfilename) = os.path.split(image_path)
            filename, _ = os.path.splitext(tempfilename)

            mask_path = os.path.join(f'{root}/{opt.dataset}/tokencut/', filename + '_tokencut.jpg')
            if opt.gt:
                mask_path = os.path.join(f'{root}/{opt.dataset}/tokencut/', filename + '_gt.jpg')
            init_mask = np.array(pad_image(Image.open(mask_path).convert("L"), (512, 512), type='L'))
            init_mask = init_mask.astype(np.float32) / 255.0
            init_mask = init_mask[None, None, ...]
            init_mask[init_mask < 0.5] = 0
            init_mask[init_mask >= 0.5] = 1
            points = None

            if opt.box_init:
                index = np.where(init_mask == 1)
                a, aa, b, bb = np.min(index[2]), np.max(index[2]), np.min(index[3]), np.max(index[3])
                mask = np.zeros_like(init_mask)
                mask[:, :, a:aa, b:bb] = 1
                init_mask = torch.from_numpy(mask).to(device)
            elif opt.scribble_init:
                mask = generate_scribble(init_mask[0, 0], num_points=2)[None, None, ...]
                dilate = get_dilate(torch.from_numpy(mask).to(device), device, iteration=1, kernel_size=5)
                init_mask = dilate.float()
            elif opt.point_num != 0:
                indices = np.argwhere(init_mask[0, 0] == 1)
                if opt.point_num != 1:
                    random_indices = np.random.choice(indices.shape[0], opt.point_num, replace=False)
                    points = indices[random_indices]
                else:
                    points = [(np.mean(indices[0]), np.mean(indices[1])),]
                mask = np.ones_like(init_mask)
                init_mask = torch.from_numpy(mask).to(device)
            else:
                init_mask = torch.from_numpy(init_mask).to(device)

            with model.ema_scope():
                # predict mask using dino feat
                save_path = image_path.replace(f'{opt.dataset}', f'PaintSeg/{opt.dataset}').replace('images', f'{opt.outdir}')
                if os.path.exists(os.path.join(save_path, 'ema.png')):
                    continue
                os.makedirs(save_path, exist_ok=True)
                mask = predict_mask(opt.iters, backbone, model, pipe, image, init_mask, dinoTransform, save_path,
                                    use_crf=opt.use_crf, global_cluster=False, points=points)
                visualize(mask, os.path.join(save_path, 'ema.png'))


