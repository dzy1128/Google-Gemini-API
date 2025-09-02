
# -*- coding: utf-8 -*-
"""
ComfyUI Custom Node: Gemini Generate & Edit Image Nodes

包含两个节点:

1. Gemini Generate Content (3-Image) - Multi-Candidate Output
功能:
- 接收 1~3 张图片（第1张必需，另外2张可选），和一个文本 prompt。
- 调用 Google AI 的 genai SDK（与用户示例一致: from google import genai），
  使用模型 "gemini-2.5-flash-image-preview" 进行多模态生成。
- 支持生成多个候选结果，解析返回的所有候选，输出：
  1) IMAGES: 生成的图片批次（来自所有候选的图片）；如果没有生成图片则回传输入的第1张图片。
  2) STRING: 合并的文本输出（来自所有候选）。
  3) BOOLEAN: 是否生成了新图片的标记。
  4) INT: 生成的图片数量。

2. Gemini Edit Image (5-Image) - Image Editing Node
功能:
- 接收 1~5 张图片（第1张必需，其余4张可选），和一个文本 prompt。
- 使用 OpenAI 兼容的 API 接口，默认使用 "https://www.chataiapi.com/v1/chat/completions"
- 使用模型 "gemini-2.5-flash-image-preview" 进行图片编辑。
- 支持 seed 参数控制生成的随机性（最小值为0）。
- 支持输出多张图片（1-4张）。
- 输出：
  1) IMAGES: 编辑后的图片批次
  2) STRING: API返回的文本说明
  3) BOOLEAN: 是否生成了新图片的标记
  4) INT: 生成的图片数量

多候选处理特性:
- 通过 candidate_count/num_outputs 参数控制生成候选数量（1-4个）
- 自动收集所有候选结果中的图片和文本
- 将所有候选的图片组合成一个ComfyUI批次张量
- 自动调整所有图片到相同尺寸（使用最大尺寸）
- 文本输出会标注来自哪个候选（当有多个候选时）
- 返回图片数量信息，便于后续处理

依赖:
- pip install google-genai openai Pillow numpy torch
- 需配置 GOOGLE_API_KEY 环境变量。

安装:
- 将本文件放入 ComfyUI/custom_nodes/ 目录下，重启 ComfyUI。

注意:
- 本节点仅取每个 IMAGE 输入的 batch 第 0 张进行处理。
- 输出的图片批次中，所有图片都会被调整到相同尺寸以保证张量一致性。
- 更多候选数量会消耗更多API配额，请根据需要调整。
"""

import os
import base64
import random
import requests
import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image

try:
    from google import genai
except Exception as e:
    genai = None

try:
    from openai import OpenAI
except ImportError as e:
    OpenAI = None


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


def _pil_to_base64(pil_img, format="JPEG"):
    """将 PIL.Image 转换为 base64 编码字符串。"""
    buffer = BytesIO()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffer, format=format)
    img_data = buffer.getvalue()
    return base64.b64encode(img_data).decode('utf-8')


class GeminiEditImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "把这张房间照片改成北欧极简风，柔和自然光，保留原有布局，墙面改浅灰，木地板更浅；输出适合海报的构图。"
                }),
                "image1": ("IMAGE", {}),  # 必填
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1
                }),
                "num_outputs": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "生成图片的数量（1-4张）"
                }),
            },
            "optional": {
                "image2": ("IMAGE", {}),  # 可选
                "image3": ("IMAGE", {}),  # 可选  
                "image4": ("IMAGE", {}),  # 可选
                "image5": ("IMAGE", {}),  # 可选
                "model_name": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
                "api_url": ("STRING", {"default": "https://www.chataiapi.com/v1/chat/completions"}),
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
    FUNCTION = "edit_images"
    CATEGORY = "Google/GenAI"
    OUTPUT_NODE = False

    def _get_openai_client(self, api_url):
        """创建OpenAI兼容的客户端"""
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed. Please run: pip install openai")
        
        # 从环境变量获取API密钥
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required but not set")
        
        return OpenAI(
            api_key=api_key,
            base_url=api_url
        )

    def edit_images(self, prompt, image1, seed, num_outputs=1, image2=None, image3=None, image4=None, image5=None, 
                   model_name="gemini-2.5-flash-image-preview", api_url="https://www.chataiapi.com/v1/chat/completions", max_retries=2):
        
        # 准备图片列表
        images = [image1]
        if image2 is not None:
            images.append(image2)
        if image3 is not None:
            images.append(image3)
        if image4 is not None:
            images.append(image4)
        if image5 is not None:
            images.append(image5)

        try:
            # 转换图片为base64格式
            content_list = [{"type": "text", "text": prompt}]
            
            for i, img_tensor in enumerate(images):
                pil_img = _comfy_image_to_pil(img_tensor)
                base64_img = _pil_to_base64(pil_img, format="JPEG")
                content_list.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })
                
        except Exception as e:
            return image1, f"[Gemini Edit Node] Failed to convert input images: {e}", False, 1

        # API调用
        client = None
        generated_images = []
        texts = []
        last_error = None
        
        # 生成指定数量的图片
        for output_idx in range(num_outputs):
            for attempt in range(max_retries + 1):
                try:
                    if client is None:
                        client = self._get_openai_client(api_url)
                    
                    # 构建请求消息
                    messages = [{
                        "role": "user",
                        "content": content_list
                    }]
                    
                    # 添加seed到请求中
                    extra_params = {}
                    if seed > 0:
                        extra_params["seed"] = seed + output_idx  # 每个输出使用不同的seed
                    
                    # 根据API URL决定使用哪种方式
                    if "chataiapi.com" in api_url:
                        # 对于自定义API，使用chat接口
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=4000,
                            **extra_params
                        )
                        
                        # 解析chat接口响应
                        if response.choices and len(response.choices) > 0:
                            choice = response.choices[0]
                            if hasattr(choice, 'message') and choice.message:
                                message_content = choice.message.content
                                
                                # 检查是否包含图片数据（base64编码）
                                if message_content and ("data:image" in message_content or "base64" in message_content):
                                    # 尝试提取base64图片数据
                                    try:
                                        # 查找base64图片数据
                                        import re
                                        base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                                        matches = re.findall(base64_pattern, message_content)
                                        
                                        if matches:
                                            for match in matches:
                                                try:
                                                    img_data = base64.b64decode(match)
                                                    pil_img = Image.open(BytesIO(img_data))
                                                    generated_images.append(pil_img)
                                                except Exception as img_e:
                                                    texts.append(f"[输出 {output_idx + 1}] 解析图片失败: {img_e}")
                                        else:
                                            # 如果没有找到base64图片，添加文本响应
                                            texts.append(f"[输出 {output_idx + 1}] {message_content}")
                                    except Exception as parse_e:
                                        texts.append(f"[输出 {output_idx + 1}] 解析响应失败: {parse_e}")
                                else:
                                    # 纯文本响应
                                    if message_content:
                                        texts.append(f"[输出 {output_idx + 1}] {message_content}")
                            else:
                                texts.append(f"[输出 {output_idx + 1}] API返回空响应")
                    else:
                        # 对于其他API，可能需要不同的处理方式
                        # 这里添加一个通用的处理方式
                        try:
                            response = client.images.generate(
                                prompt=prompt,
                                model=model_name,
                                n=1,
                                size="1024x1024",
                                response_format="b64_json",
                                **extra_params
                            )
                            
                            # 解析图片生成响应
                            if hasattr(response, 'data') and response.data:
                                for img_data in response.data:
                                    if hasattr(img_data, 'b64_json'):
                                        try:
                                            img_bytes = base64.b64decode(img_data.b64_json)
                                            pil_img = Image.open(BytesIO(img_bytes))
                                            generated_images.append(pil_img)
                                        except Exception as img_e:
                                            texts.append(f"[输出 {output_idx + 1}] 解析图片失败: {img_e}")
                        except Exception as alt_e:
                            # 如果图片生成API不可用，回退到chat方式
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                max_tokens=4000,
                                **extra_params
                            )
                            
                            if response.choices and len(response.choices) > 0:
                                choice = response.choices[0]
                                if hasattr(choice, 'message') and choice.message:
                                    message_content = choice.message.content
                                    if message_content:
                                        texts.append(f"[输出 {output_idx + 1}] {message_content}")
                                else:
                                    texts.append(f"[输出 {output_idx + 1}] API返回空响应")
                    break  # 成功，退出重试循环
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # 对于某些错误，不进行重试
                    if ("400" in error_str or "429" in error_str or "403" in error_str):
                        break
                    
                    # 对于可重试的错误，等待后重试
                    if attempt < max_retries:
                        import time
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
            
            # 如果这次调用失败，记录错误但继续尝试下一个输出
            if last_error is not None:
                texts.append(f"[输出 {output_idx + 1}] API调用失败: {last_error}")
                last_error = None  # 重置错误以便下次调用

        # 处理最终结果
        image_generated = len(generated_images) > 0
        
        if generated_images:
            # 如果有生成的图片，转换为ComfyUI格式
            out_tensor = _pil_list_to_comfy_images(generated_images)
            image_count = len(generated_images)
        else:
            # 未生成图片则回传输入的第1张
            out_tensor = image1
            image_count = 1
            if not texts:
                texts.append("未能生成任何图片，请检查API配置和网络连接")

        text_out = "\n".join([t for t in texts if t])
        return out_tensor, text_out, image_generated, image_count


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
                    "min": 0,
                    "max": 2147483647,  # INT32 最大值，与API兼容
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

        # API 调用重试逻辑 - 支持多次调用获得多候选
        client = None
        responses = []  # 存储所有响应
        last_error = None
        
        # 尝试获取多个候选结果
        for candidate_idx in range(candidate_count):
            resp = None
            for attempt in range(max_retries + 1):
                try:
                    if client is None:
                        client = self._get_client()
                    
                    # 尝试不同的参数传递方式
                    try:
                        # 方法1: 直接传递candidate_count参数
                        resp = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            candidate_count=candidate_count,
                        )
                        # 如果成功且返回多个候选，直接使用这个响应
                        if hasattr(resp, 'candidates') and len(resp.candidates) >= candidate_count:
                            responses = [resp]  # 使用这个包含多候选的响应
                            break  # 退出候选循环
                    except TypeError as e1:
                        try:
                            # 方法2: 使用generation_config字典
                            generation_config = {
                                "candidate_count": candidate_count,
                            }
                            resp = client.models.generate_content(
                                model=model_name,
                                contents=contents,
                                generation_config=generation_config,
                            )
                            # 如果成功且返回多个候选，直接使用这个响应
                            if hasattr(resp, 'candidates') and len(resp.candidates) >= candidate_count:
                                responses = [resp]  # 使用这个包含多候选的响应
                                break  # 退出候选循环
                        except TypeError as e2:
                            # 方法3: 回退到单次调用
                            if candidate_idx == 0:  # 只在第一次警告
                                print(f"[Gemini Node] Warning: API不支持candidate_count参数，将通过多次调用获得{candidate_count}个结果")
                            resp = client.models.generate_content(
                                model=model_name,
                                contents=contents,
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
            
            # 如果获得了响应，添加到列表中
            if resp is not None:
                responses.append(resp)
                # 如果这个响应包含多个候选，就不需要继续循环了
                if hasattr(resp, 'candidates') and len(resp.candidates) > 1:
                    break
            else:
                # 如果这次调用失败，停止尝试更多候选
                break
        
        # 处理最终失败的情况
        if not responses:
            error_str = str(last_error) if last_error else "Unknown error"
            
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
            # 安全地解析所有响应
            total_candidates = 0
            for resp_idx, resp in enumerate(responses):
                # 检查响应结构
                if not hasattr(resp, 'candidates') or not resp.candidates:
                    texts.append(f"[响应 {resp_idx + 1}] API 返回空响应，没有生成内容")
                    continue
                
                # 遍历这个响应中的所有候选结果
                for candidate_idx, candidate in enumerate(resp.candidates):
                    total_candidates += 1
                    # 当有多个响应或多个候选时显示标识
                    if len(responses) > 1 or len(resp.candidates) > 1:
                        candidate_prefix = f"[候选 {total_candidates}] "
                    else:
                        candidate_prefix = ""
                    
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
                        if total_candidates == 1 and not texts:  # 只为第一个候选添加错误信息
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
    "GeminiEditImage": GeminiEditImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiGenerate": "Gemini Generate",
    "GeminiEditImage": "Gemini Edit Image",
}

