import os, shutil, requests, random, time, uuid, boto3, runpod, gc
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime

import torch
import numpy as np
from PIL import Image, ImageOps

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

os.environ["HF_HOME"] = "/content/cache"

from torch.amp import autocast

from typing import Tuple
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

from transformers import AutoModelForImageSegmentation
from torchvision import transforms

birefnet = AutoModelForImageSegmentation.from_pretrained("camenduru/RMBG-2.0", trust_remote_code=True)
birefnet.to("cuda")
birefnet.eval()

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.inference_mode()
def remove_background_local(image_pil: Image.Image) -> Image.Image:
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    original_size = image_pil.size
    w, h = image_pil.size
    new_w = ((w + 31) // 32) * 32
    new_h = ((h + 31) // 32) * 32
    pad_w = new_w - w
    pad_h = new_h - h
    image_pil = ImageOps.expand(image_pil, (0, 0, pad_w, pad_h), fill=0)
    input_tensor = transform_image(image_pil).unsqueeze(0).to("cuda")
    with autocast("cuda", dtype=torch.float16):
        pred = birefnet(input_tensor)[-1].sigmoid().cpu()[0].squeeze()
    mask = transforms.ToPILImage()(pred)
    mask = mask.crop((0, 0, original_size[0], original_size[1]))
    image_pil = image_pil.crop((0, 0, original_size[0], original_size[1]))
    image_pil.putalpha(mask)
    return image_pil

pipeline = Trellis2ImageTo3DPipeline.from_pretrained('camenduru/TRELLIS.2-4B')
pipeline.rembg_model = None
pipeline.low_vram = True
pipeline.cuda()

TMP_DIR = "/content/TRELLIS.2/tmp"
os.makedirs(TMP_DIR, exist_ok=True)
MAX_SEED = np.iinfo(np.int32).max

@torch.inference_mode()
def preprocess_image(input: Image.Image, remove_bg: bool = True) -> Image.Image:
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    if remove_bg:
        input = remove_background_local(input)
    arr = np.array(input)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
    else:
        alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
        input = input.convert("RGBA")  # Ensure RGBA for consistency
        arr = np.array(input)
    coords = np.argwhere(alpha > 200)
    if len(coords) == 0:
        cropped = input
    else:
        y0, x0 = coords.min(axis=0)[:2]
        y1, x1 = coords.max(axis=0)[:2]
        cropped = input.crop((x0, y0, x1 + 1, y1 + 1))
    width, height = cropped.size
    max_dim = max(width, height)
    squared = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
    pad_left = (max_dim - width) // 2
    pad_top = (max_dim - height) // 2
    squared.paste(cropped, (pad_left, pad_top))
    arr = np.array(squared).astype(np.float32) / 255.0
    arr[:, :, :3] *= arr[:, :, 3:4]  # RGB *= alpha
    final_image = Image.fromarray((arr * 255).astype(np.uint8))
    return final_image

def pack_state(latents: Tuple[SparseTensor, SparseTensor, int]) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }

def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']

@torch.inference_mode()
def image_to_3d(
    image: Image.Image,
    seed: int = 42,
    resolution: str = "1024",
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    remove_bg: bool = True,
) -> dict:
    processed_image = preprocess_image(image, remove_bg)
  
    # Clear cache before running the pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    outputs, latents = pipeline.run(
        processed_image,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution],
        return_latent=True,
    )
 
    state = pack_state(latents)

    del outputs, latents, processed_image
    torch.cuda.empty_cache()
    gc.collect()

    return state

def extract_glb(
    state: dict,
    decimation_target: int = 300000,
    texture_size: int = 2048,
    output_dir: str = TMP_DIR
) -> str:
    gc.collect()
    torch.cuda.empty_cache()
    
    shape_slat, tex_slat, res = unpack_state(state)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    mesh.simplify(16777216)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
     
    del shape_slat, tex_slat, res, mesh
    pipeline.current_latents = None
    pipeline.decoded_mesh = None

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"trellis2_{timestamp}.glb")
    glb.export(glb_path, extension_webp=True)

    del glb
    torch.cuda.empty_cache()
    gc.collect()
    return glb_path


@torch.inference_mode()
def generate(input):
    try:
        tmp_dir="/content/TRELLIS.2/tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:6]
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        s3_access_key_id = os.getenv('s3_access_key_id')
        s3_secret_access_key = os.getenv('s3_secret_access_key')
        s3_endpoint_url = os.getenv('s3_endpoint_url')
        s3_region_name = os.getenv('s3_region_name')
        s3_bucket_name = os.getenv('s3_bucket_name')
        s3_bucket_folder = os.getenv('s3_bucket_folder')
        s3 = boto3.client('s3', aws_access_key_id=s3_access_key_id, aws_secret_access_key=s3_secret_access_key, endpoint_url=s3_endpoint_url, region_name=s3_region_name)

        values = input["input"]
        job_id = values['job_id']

        input_image = values['input_image']
        input_image = download_file(url=input_image, save_dir=tmp_dir, file_name='input_image')
        seed = values['seed'] # 0
        resolution = values['resolution'] # 1024
        ss_guidance_strength = values['ss_guidance_strength'] # 7.5,
        ss_guidance_rescale = values['ss_guidance_rescale'] # 0.7,
        ss_sampling_steps = values['ss_sampling_steps'] # 12,
        ss_rescale_t = values['ss_rescale_t'] # 5.0,
        shape_slat_guidance_strength = values['shape_slat_guidance_strength'] # 7.5,
        shape_slat_guidance_rescale = values['shape_slat_guidance_rescale'] # 0.5,
        shape_slat_sampling_steps = values['shape_slat_sampling_steps'] # 12,
        shape_slat_rescale_t = values['shape_slat_rescale_t'] # 3.0,
        tex_slat_guidance_strength = values['tex_slat_guidance_strength'] # 1.0,
        tex_slat_guidance_rescale = values['tex_slat_guidance_rescale'] # 0.0,
        tex_slat_sampling_steps = values['tex_slat_sampling_steps'] # 12,
        tex_slat_rescale_t = values['tex_slat_rescale_t'] # 3.0,
        remove_bg = values['remove_bg'] # True

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, MAX_SEED)

        img = Image.open(input_image).convert("RGBA")
        state = image_to_3d(img, seed=seed, resolution=resolution,ss_guidance_strength=ss_guidance_strength,ss_guidance_rescale=ss_guidance_rescale,
                            ss_sampling_steps=ss_sampling_steps,ss_rescale_t=ss_rescale_t,shape_slat_guidance_strength=shape_slat_guidance_strength,
                            shape_slat_guidance_rescale=shape_slat_guidance_rescale,shape_slat_sampling_steps=shape_slat_sampling_steps,
                            shape_slat_rescale_t=shape_slat_rescale_t,tex_slat_guidance_strength=tex_slat_guidance_strength,
                            tex_slat_guidance_rescale=tex_slat_guidance_rescale,tex_slat_sampling_steps=tex_slat_sampling_steps,tex_slat_rescale_t=tex_slat_rescale_t,remove_bg=remove_bg)
        glb_path = extract_glb(state, decimation_target=300000, texture_size=2048)

        result = glb_path

        s3_key =  f"{s3_bucket_folder}/trellis-2-{current_time}-{seed}-{unique_id}.glb"
        s3.upload_file(result, s3_bucket_name, s3_key)
        result_url = f"{s3_endpoint_url}/{s3_bucket_name}/{s3_key}"

        return {"job_id": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        return {"job_id": job_id, "result": str(e), "status": "FAILED"}
    finally:
        if 'shape_slat' in locals():
            del shape_slat
        if 'tex_slat' in locals():
            del tex_slat
        if 'mesh' in locals():
            del mesh
        if hasattr(pipeline, 'current_latents'):
            pipeline.current_latents = None
        if hasattr(pipeline, 'decoded_mesh'):
            pipeline.decoded_mesh = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        directory_path = Path(tmp_dir)
        if directory_path.exists():
            shutil.rmtree(directory_path)
            print(f"Directory {directory_path} has been removed successfully.")
        else:
            print(f"Directory {directory_path} does not exist.")

runpod.serverless.start({"handler": generate})