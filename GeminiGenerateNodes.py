
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
                    "default": "Generate an image of a cute cat eating a nano-banana in a fancy restaurant under the gemini constellation. Create the image, don't just describe it."
                }),
                "image1": ("IMAGE", {}),  # 必填
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,  # 64位有符号整数最小值
                    "max": 0xffffffffffffffff,   # 64位有符号整数最大值
                    "step": 1
                }),
            },
            "optional": {
                "image2": ("IMAGE", {}),  # 可选
                "image3": ("IMAGE", {}),  # 可选
                "model_name": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
                "max_retries": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1
                }),
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
        # 按照官方示例使用默认客户端创建方式
        # 环境变量会自动被SDK使用
        return genai.Client()

    def generate(self, prompt, image1, seed, image2=None, image3=None, model_name="gemini-2.5-flash-image-preview", max_retries=2):
        # 构造 contents: [prompt, image1, (image2?), (image3?)]
        # seed 参数只用于 ComfyUI 节点，不传递给 Gemini API
        contents = [str(prompt)]

        try:
            pil1 = _comfy_image_to_pil(image1)
            contents.append(pil1)
            if image2 is not None:
                contents.append(_comfy_image_to_pil(image2))
            if image3 is not None:
                contents.append(_comfy_image_to_pil(image3))
        except Exception as e:
            # 转换失败，直接返回错误文本和原图
            return image1, f"[Gemini Node] Failed to convert input images: {e}", False

        # API 调用重试逻辑
        client = None
        resp = None
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if client is None:
                    print(f"[DEBUG] Creating client...")
                    client = self._get_client()
                
                print(f"[DEBUG] API call attempt {attempt + 1}/{max_retries + 1} with {len(contents)} contents: 1 prompt + {len(contents)-1} images")
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                print(f"[DEBUG] API call successful on attempt {attempt + 1}")
                break  # 成功，退出重试循环
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                print(f"[DEBUG] API call attempt {attempt + 1} failed: {e}")
                
                # 对于某些错误，不进行重试
                if ("400" in error_str or "FAILED_PRECONDITION" in error_str or 
                    "User location is not supported" in error_str or
                    "RESOURCE_EXHAUSTED" in error_str):
                    print(f"[DEBUG] Non-retryable error, stopping retries")
                    break
                
                # 对于可重试的错误（如 500），等待后重试
                if attempt < max_retries:
                    import time
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2秒, 4秒, 6秒...
                    print(f"[DEBUG] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        # 处理最终失败的情况
        if resp is None:
            error_str = str(last_error)
            print(f"[DEBUG] All API call attempts failed. Last error: {last_error}")
            
            # 检查具体的错误类型并提供相应的建议
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                return image1, f"[Gemini Node] API配额已用完，请稍后重试或升级计划。", False
            elif "400" in error_str and "User location is not supported" in error_str:
                return image1, f"[Gemini Node] 地理位置限制：你所在的地区不支持使用 Gemini API。请检查代理设置。", False
            elif "500" in error_str or "INTERNAL" in error_str:
                return image1, f"[Gemini Node] Google 服务器内部错误 (500)，已重试 {max_retries} 次仍失败，请稍后再试。", False
            elif "503" in error_str or "SERVICE_UNAVAILABLE" in error_str:
                return image1, f"[Gemini Node] 服务暂时不可用 (503)，已重试 {max_retries} 次仍失败，请稍后再试。", False
            elif "FAILED_PRECONDITION" in error_str:
                return image1, f"[Gemini Node] API 使用条件不满足，可能是地理位置限制或其他条件问题。", False
            else:
                return image1, f"[Gemini Node] API call failed after {max_retries + 1} attempts: {last_error}", False

        texts = []
        out_tensor = None
        image_generated = False

        try:
            # 完全按照官方示例的方式解析响应
            print(f"[DEBUG] Parsing response with {len(resp.candidates)} candidates")
            parts = resp.candidates[0].content.parts
            print(f"[DEBUG] Found {len(parts)} parts in response")
            
            for i, part in enumerate(parts):
                print(f"[DEBUG] Part {i}: text={part.text is not None}, inline_data={part.inline_data is not None}")
                if part.text is not None:
                    texts.append(part.text)
                    print(f"[DEBUG] Added text: {part.text[:200]}...")
                    # 检查文本是否包含拒绝生成图像的信息
                    if any(keyword in part.text.lower() for keyword in ['cannot generate', 'cannot create', 'unable to generate', 'unable to create']):
                        print(f"[DEBUG] Model refused to generate image")
                    else:
                        print(f"[DEBUG] Model returned text instead of image")
                elif part.inline_data is not None:
                    try:
                        print(f"[DEBUG] Processing generated image data...")
                        pil = Image.open(BytesIO(part.inline_data.data))
                        print(f"[DEBUG] Generated image size: {pil.size}")
                        out_tensor = _pil_to_comfy_image(pil)
                        image_generated = True
                        print(f"[DEBUG] Successfully converted generated image to tensor")
                    except Exception as img_e:
                        print(f"[DEBUG] Failed to process generated image: {img_e}")
                        texts.append(f"[Gemini Node] Failed to process generated image: {img_e}")
        except Exception as e:
            print(f"[DEBUG] Failed to parse response: {e}")
            texts.append(f"[Gemini Node] Failed to parse response: {e}")

        if out_tensor is None:
            out_tensor = image1  # 未生成图片则回传输入的第1张
            image_generated = False

        text_out = "\n".join([t for t in texts if t])
        print(f"[DEBUG] Final result: image_generated={image_generated}, text_length={len(text_out)}")
        return out_tensor, text_out, image_generated


NODE_CLASS_MAPPINGS = {
    "GeminiGenerate": GeminiGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiGenerate": "Gemini Generate",
}

