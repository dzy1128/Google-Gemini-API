
# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Gemini Generate Content (3-Image) - Multi-Candidate Output

功能:
- 接收 1~3 张图片（第1张必需，另外2张可选），和一个文本 prompt。
- 调用 Google AI 的 genai SDK（与用户示例一致: from google import genai），
  使用模型 "gemini-2.5-flash-image-preview" 进行多模态生成。
- 支持生成多个候选结果，解析返回的所有候选，输出：
  1) IMAGES: 生成的图片批次（来自所有候选的图片）；如果没有生成图片则回传输入的第1张图片。
  2) STRING: 合并的文本输出（来自所有候选）。
  3) BOOLEAN: 是否生成了新图片的标记。
  4) INT: 生成的图片数量。

多候选处理特性:
- 通过 candidate_count 参数控制生成候选数量（1-4个）
- 自动收集所有候选结果中的图片和文本
- 将所有候选的图片组合成一个ComfyUI批次张量
- 自动调整所有图片到相同尺寸（使用最大尺寸）
- 文本输出会标注来自哪个候选（当有多个候选时）
- 返回图片数量信息，便于后续处理

候选数量说明:
- candidate_count=1: 生成1个候选结果（默认）
- candidate_count=2-4: 生成多个候选结果，增加图片多样性

依赖:
- pip install google-genai Pillow numpy torch
- 需配置 GOOGLE_API_KEY 环境变量。

安装:
- 将本文件放入 ComfyUI/custom_nodes/ 目录下，重启 ComfyUI。

注意:
- 本节点仅取每个 IMAGE 输入的 batch 第 0 张进行处理。
- 输出的图片批次中，所有图片都会被调整到相同尺寸以保证张量一致性。
- 更多候选数量会消耗更多API配额，请根据需要调整。
- API调用: client.models.generate_content(model=..., contents=[...], generation_config={"candidate_count": N})。
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


def _pil_list_to_comfy_images(pil_list):
    """将 PIL.Image 列表转为 ComfyUI 的 IMAGE tensor ([N,H,W,3], float32, 0..1)。"""
    if not pil_list:
        return None
    
    # 转换所有图片并收集尺寸信息
    arrays = []
    max_height = 0
    max_width = 0
    
    for pil_img in pil_list:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        
        arrays.append(arr)
        max_height = max(max_height, arr.shape[0])
        max_width = max(max_width, arr.shape[1])
    
    # 将所有图片调整到相同尺寸（使用最大尺寸）
    normalized_arrays = []
    for arr in arrays:
        if arr.shape[0] != max_height or arr.shape[1] != max_width:
            # 将数组转回PIL图片进行尺寸调整
            temp_pil = Image.fromarray((arr * 255.0).astype(np.uint8))
            temp_pil = temp_pil.resize((max_width, max_height), Image.Resampling.LANCZOS)
            arr = np.array(temp_pil).astype(np.float32) / 255.0
        normalized_arrays.append(arr)
    
    # 堆叠成批次张量
    batch_tensor = torch.from_numpy(np.stack(normalized_arrays, axis=0))
    return batch_tensor


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
                "candidate_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "生成候选图片的数量（1-4张）"
                }),
                "model_name": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
                "max_retries": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("images", "text", "image_generated", "image_count")
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

    def generate(self, prompt, image1, seed, image2=None, image3=None, candidate_count=1, model_name="gemini-2.5-flash-image-preview", max_retries=2):
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
            return image1, f"[Gemini Node] Failed to convert input images: {e}", False, 1

        # API 调用重试逻辑
        client = None
        resp = None
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if client is None:
                    client = self._get_client()
                
                # 配置生成参数
                generation_config = {
                    "candidate_count": candidate_count,
                }
                
                resp = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    generation_config=generation_config,
                )
                break  # 成功，退出重试循环
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 对于某些错误，不进行重试
                if ("400" in error_str or "FAILED_PRECONDITION" in error_str or 
                    "User location is not supported" in error_str or
                    "RESOURCE_EXHAUSTED" in error_str or
                    "PROHIBITED_CONTENT" in error_str):
                    break
                
                # 对于可重试的错误（如 500），等待后重试
                if attempt < max_retries:
                    import time
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2秒, 4秒, 6秒...
                    time.sleep(wait_time)
        
        # 处理最终失败的情况
        if resp is None:
            error_str = str(last_error)
            
            # 检查具体的错误类型并提供相应的建议
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                return image1, f"[Gemini Node] API配额已用完，请稍后重试或升级计划。", False, 1
            elif "400" in error_str and "User location is not supported" in error_str:
                return image1, f"[Gemini Node] 地理位置限制：你所在的地区不支持使用 Gemini API。请检查代理设置。", False, 1
            elif "PROHIBITED_CONTENT" in error_str:
                return image1, f"[Gemini Node] 内容被拒绝：模型认为请求包含禁止的内容。请尝试修改提示词或图像。", False, 1
            elif "500" in error_str or "INTERNAL" in error_str:
                return image1, f"[Gemini Node] Google 服务器内部错误 (500)，已重试 {max_retries} 次仍失败，请稍后再试。", False, 1
            elif "503" in error_str or "SERVICE_UNAVAILABLE" in error_str:
                return image1, f"[Gemini Node] 服务暂时不可用 (503)，已重试 {max_retries} 次仍失败，请稍后再试。", False, 1
            elif "FAILED_PRECONDITION" in error_str:
                return image1, f"[Gemini Node] API 使用条件不满足，可能是地理位置限制或其他条件问题。", False, 1
            else:
                return image1, f"[Gemini Node] API call failed after {max_retries + 1} attempts: {last_error}", False, 1

        texts = []
        generated_images = []  # 收集所有生成的图片
        image_generated = False

        try:
            # 安全地解析响应
            # 检查响应结构
            if not hasattr(resp, 'candidates') or not resp.candidates:
                texts.append("[Gemini Node] API 返回空响应，没有生成内容")
            else:
                # 遍历所有候选结果
                for candidate_idx, candidate in enumerate(resp.candidates):
                    candidate_prefix = f"[候选 {candidate_idx + 1}] " if len(resp.candidates) > 1 else ""
                    
                    # 检查是否有 finish_reason，这能告诉我们为什么没有内容
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        finish_reason = str(candidate.finish_reason)
                        
                        if 'PROHIBITED_CONTENT' in finish_reason:
                            texts.append(f"{candidate_prefix}内容被拒绝：模型认为请求包含禁止的内容。")
                        elif 'SAFETY' in finish_reason:
                            texts.append(f"{candidate_prefix}安全检查失败：请求触发了安全过滤器。")
                        elif 'MAX_TOKENS' in finish_reason:
                            texts.append(f"{candidate_prefix}达到最大 token 限制，响应被截断。")
                        elif 'STOP' in finish_reason:
                            pass  # 正常完成，继续处理内容
                        else:
                            texts.append(f"{candidate_prefix}生成因未知原因停止：{finish_reason}")
                    
                    # 检查 content 是否存在
                    if not hasattr(candidate, 'content') or candidate.content is None:
                        if candidate_idx == 0 and not texts:  # 只为第一个候选添加错误信息
                            texts.append(f"{candidate_prefix}API 响应中没有内容数据")
                    else:
                        content = candidate.content
                        
                        # 检查 parts 是否存在
                        if not hasattr(content, 'parts') or not content.parts:
                            texts.append(f"{candidate_prefix}响应内容为空")
                        else:
                            parts = content.parts
                            
                            for part in parts:
                                if part.text is not None:
                                    text_with_prefix = f"{candidate_prefix}{part.text}" if candidate_prefix else part.text
                                    texts.append(text_with_prefix)
                                elif part.inline_data is not None:
                                    try:
                                        pil = Image.open(BytesIO(part.inline_data.data))
                                        generated_images.append(pil)
                                        image_generated = True
                                    except Exception as img_e:
                                        texts.append(f"{candidate_prefix}Failed to process generated image: {img_e}")
                
        except Exception as e:
            texts.append(f"[Gemini Node] Failed to parse response: {e}")

        # 处理图片输出
        if generated_images:
            # 如果有生成的图片，转换为ComfyUI格式
            out_tensor = _pil_list_to_comfy_images(generated_images)
            image_count = len(generated_images)
        else:
            # 未生成图片则回传输入的第1张
            out_tensor = image1
            image_generated = False
            image_count = 1

        text_out = "\n".join([t for t in texts if t])
        return out_tensor, text_out, image_generated, image_count


NODE_CLASS_MAPPINGS = {
    "GeminiGenerate": GeminiGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiGenerate": "Gemini Generate",
}

