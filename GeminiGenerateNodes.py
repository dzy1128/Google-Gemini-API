
# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Gemini Generate Content (3-Image)

功能:
- 接收 1~3 张图片（第1张必需，另外2张可选），和一个文本 prompt。
- 调用 Google AI 的 genai SDK（与用户示例一致: from google import genai），
  使用模型 "gemini-2.5-flash-image-preview" 进行多模态生成。
- 解析返回的候选结果，输出：
  1) IMAGE: 如果返回包含图片，则输出生成图片；否则回传输入的第1张图片。
  2) STRING: 合并的文本输出（若有）。
  3) BOOLEAN: 是否生成了新图片的标记。

依赖:
- pip install google-genai Pillow numpy torch
- 需配置 GOOGLE_API_KEY 环境变量。

安装:
- 将本文件放入 ComfyUI/custom_nodes/ 目录下，重启 ComfyUI。

注意:
- 本节点仅取每个 IMAGE 输入的 batch 第 0 张进行处理。
- 严格遵从用户提供的示例用法: client.models.generate_content(model=..., contents=[prompt, image1, image2?, image3?])。
"""

import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image

try:
    from google import genai
except Exception as e:
    genai = None


def _comfy_image_to_pil(img_tensor):
    """将 ComfyUI 的 IMAGE (tensor, [B,H,W,C], float32, 0..1) 转为单张 PIL.Image。
    仅取批次的第 0 张。
    """
    if isinstance(img_tensor, torch.Tensor):
        arr = img_tensor[0].detach().cpu().numpy()
    else:
        # 兼容 numpy 输入
        arr = np.asarray(img_tensor)[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    # 只保留前3通道
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr)


def _pil_to_comfy_image(pil_img):
    """将 PIL.Image 转为 ComfyUI 的 IMAGE tensor ([1,H,W,3], float32, 0..1)。"""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    tensor = torch.from_numpy(arr)[None, ...]
    return tensor


class GeminiGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Create a picture of my cat eating a nano-banana in a fancy restaurant under the gemini constellation"
                }),
                "image1": ("IMAGE", {}),  # 必填
            },
            "optional": {
                "image2": ("IMAGE", {}),  # 可选
                "image3": ("IMAGE", {}),  # 可选
                "model_name": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("image", "text", "image_generated")
    FUNCTION = "generate"
    CATEGORY = "Google/GenAI"
    OUTPUT_NODE = False

    def _get_client(self):
        if genai is None:
            raise RuntimeError("google-genai SDK not installed. Please run: pip install google-genai")
        key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required but not set")
        return genai.Client(api_key=key)

    def generate(self, prompt, image1, image2=None, image3=None, model_name="gemini-2.5-flash-image-preview"):
        # 构造 contents: [prompt, image1, (image2?), (image3?)]
        contents = [str(prompt)]
        
        print(f"[Gemini Debug] Starting generation with prompt: {prompt[:100]}...")
        print(f"[Gemini Debug] Model: {model_name}")

        try:
            pil1 = _comfy_image_to_pil(image1)
            contents.append(pil1)
            if image2 is not None:
                contents.append(_comfy_image_to_pil(image2))
            if image3 is not None:
                contents.append(_comfy_image_to_pil(image3))
            print(f"[Gemini Debug] Successfully converted {len(contents)-1} images")
        except Exception as e:
            # 转换失败，直接返回错误文本和原图
            print(f"[Gemini Debug] Image conversion failed: {e}")
            return image1, f"[Gemini Node] Failed to convert input images: {e}", False

        try:
            print("[Gemini Debug] Getting client...")
            client = self._get_client()
            print("[Gemini Debug] Client created, making API call...")
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            print(f"[Gemini Debug] API call successful, response type: {type(resp)}")
        except Exception as e:
            print(f"[Gemini Debug] API call failed: {e}")
            return image1, f"[Gemini Node] API call failed: {e}", False

        texts = []
        out_tensor = None
        image_generated = False

        try:
            print("[Gemini Debug] Parsing response...")
            candidates = getattr(resp, "candidates", None)
            print(f"[Gemini Debug] Found {len(candidates) if candidates else 0} candidates")
            if candidates:
                parts = candidates[0].content.parts
                print(f"[Gemini Debug] Found {len(parts)} parts in first candidate")
                for i, part in enumerate(parts):
                    print(f"[Gemini Debug] Part {i}: has text={hasattr(part, 'text')}, has inline_data={hasattr(part, 'inline_data')}")
                    if getattr(part, "text", None):
                        texts.append(part.text)
                        print(f"[Gemini Debug] Added text: {part.text[:100]}...")
                    elif getattr(part, "inline_data", None) is not None and getattr(part.inline_data, "data", None) is not None:
                        try:
                            print("[Gemini Debug] Found inline image data, converting...")
                            pil = Image.open(BytesIO(part.inline_data.data))
                            out_tensor = _pil_to_comfy_image(pil)
                            image_generated = True
                            print("[Gemini Debug] Successfully converted generated image")
                        except Exception as img_e:
                            print(f"[Gemini Debug] Failed to convert generated image: {img_e}")
                            pass
        except Exception as e:
            print(f"[Gemini Debug] Failed to parse response: {e}")
            texts.append(f"[Gemini Node] Failed to parse response: {e}")

        if out_tensor is None:
            print("[Gemini Debug] No generated image found, returning original image")
            out_tensor = image1  # 未生成图片则回传输入的第1张
            image_generated = False
        else:
            print("[Gemini Debug] Returning generated image")

        text_out = "\n".join([t for t in texts if t])
        print(f"[Gemini Debug] Final text output: {text_out[:100] if text_out else 'No text'}")
        print(f"[Gemini Debug] Image generated: {image_generated}")
        return out_tensor, text_out, image_generated


NODE_CLASS_MAPPINGS = {
    "GeminiGenerate": GeminiGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiGenerate": "Gemini Generate",
}

