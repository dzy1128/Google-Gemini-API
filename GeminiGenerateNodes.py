
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
except Exception as e:
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


class OpenAIGeminiGenerate:
    """
    ComfyUI Custom Node: OpenAI 兼容格式的 Gemini 视觉理解节点（多图片支持）
    
    功能:
    - 通过 OpenAI 兼容的 API 接口调用 Gemini 视觉模型
    - 支持多张图片输入：第1张必选，第2、3张可选
    - 图片自动转换为 base64 编码
    - 从环境变量获取 API Key
    - 必需设置 seed 值（范围：0 到 2147483647，INT32 格式）
    - 默认使用 gemini-2.5-flash-image-preview（视觉理解模型）
    - 支持 reasoning_content 输出（思考过程和最终答案分离）
    
    注意:
    - gemini-2.5-flash-image-preview 是视觉理解模型，不生成新图片
    - 主要用于分析和理解输入的图片内容
    - 如需图片生成，请使用支持图片生成的模型（如 dall-e-3）
    
    输入:
    - image1: 第一张图片（必选）
    - image2: 第二张图片（可选）
    - image3: 第三张图片（可选）
    
    依赖:
    - pip install openai Pillow numpy torch
    - 需配置环境变量：OPENAI_API_KEY 或 DEEPSEEK_API_KEY
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "请详细分析这些图片的内容，包括：1. 图片中的主要物体和场景 2. 颜色、风格和构图特点 3. 可能的用途或背景信息。"
                }),
                "image1": ("IMAGE", {}),  # 第一张图片必选
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,  # INT32 最大值
                    "step": 1,
                    "tooltip": "随机种子，最小值为0，最大值为2147483647 (INT32范围)"
                }),
                "model_name": ("STRING", {
                    "default": "gemini-2.5-flash-image-preview",
                    "tooltip": "常用模型名称: gemini-2.5-flash-image-preview, gemini-pro-vision, gpt-4-vision-preview, gpt-4o"
                }),
                "api_key_env": (["OPENAI_API_KEY", "DEEPSEEK_API_KEY"], {
                    "default": "DEEPSEEK_API_KEY"
                }),
                "base_url": ("STRING", {
                    "default": "https://www.chataiapi.com/v1"
                }),
            },
            "optional": {
                "image2": ("IMAGE", {}),  # 第二张图片可选
                "image3": ("IMAGE", {}),  # 第三张图片可选
                "max_retries": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("images", "reasoning", "final_answer", "image_generated")
    FUNCTION = "generate"
    CATEGORY = "Google/GenAI"
    OUTPUT_NODE = False

    def _get_client(self, api_key_env, base_url):
        """获取 OpenAI 客户端"""
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed. Please run: pip install openai")
        
        # 从环境变量获取 API Key
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"{api_key_env} environment variable is required but not set")
        
        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, prompt, image1, seed, model_name, api_key_env="DEEPSEEK_API_KEY", 
                 base_url="https://www.chataiapi.com/v1", image2=None, image3=None, max_retries=2):
        
        # 验证 seed 值范围（API 要求 INT32 范围）
        if seed < 0 or seed > 2147483647:
            return image1, "", f"[OpenAI Gemini Node] Seed 值必须在 0 到 2147483647 范围内，当前值: {seed}", False
        
        print(f"[DEBUG] 开始处理请求 - 模型: {model_name}")
        print(f"[DEBUG] API配置 - 环境变量: {api_key_env}, 基础URL: {base_url}")
        
        try:
            # 处理第一张图片（必选）
            pil_img1 = _comfy_image_to_pil(image1)
            base64_img1 = _pil_to_base64(pil_img1)
            
            # 构建内容列表，从文本开始
            content_list = [{"type": "text", "text": str(prompt)}]
            
            # 添加第一张图片
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img1}"
                }
            })
            
            # 处理第二张图片（可选）
            if image2 is not None:
                pil_img2 = _comfy_image_to_pil(image2)
                base64_img2 = _pil_to_base64(pil_img2)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img2}"
                    }
                })
            
            # 处理第三张图片（可选）
            if image3 is not None:
                pil_img3 = _comfy_image_to_pil(image3)
                base64_img3 = _pil_to_base64(pil_img3)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img3}"
                    }
                })
            
        except Exception as e:
            return image1, "", f"[OpenAI Gemini Node] Failed to process input images: {e}", False

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        # API 调用重试逻辑
        client = None
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if client is None:
                    client = self._get_client(api_key_env, base_url)
                
                # 调用 API
                print(f"[DEBUG] 调用API - 模型: {model_name}, seed: {seed}")
                print(f"[DEBUG] 消息数量: {len(messages)}, 内容项数量: {len(messages[0]['content'])}")
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    seed=seed
                )
                
                print(f"[DEBUG] API调用成功，响应类型: {type(response)}")
                
                # 解析响应
                if not response.choices:
                    return image1, "", "[OpenAI Gemini Node] API 返回空响应", False
                
                print(f"[DEBUG] 响应包含 {len(response.choices)} 个选择")
                
                choice = response.choices[0]
                message = choice.message
                
                print(f"[DEBUG] Message 对象属性: {[attr for attr in dir(message) if not attr.startswith('_')]}")
                print(f"[DEBUG] Message content: {getattr(message, 'content', 'NO CONTENT')}")
                print(f"[DEBUG] Message reasoning_content: {getattr(message, 'reasoning_content', 'NO REASONING')}")
                
                # 提取 reasoning_content 和 content
                reasoning_content = ""
                final_answer = ""
                
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning_content = message.reasoning_content
                    print(f"[DEBUG] 找到 reasoning_content: {len(reasoning_content)} 字符")
                else:
                    print(f"[DEBUG] 没有找到 reasoning_content")
                
                if hasattr(message, 'content') and message.content:
                    final_answer = message.content
                    print(f"[DEBUG] 找到 content: {len(final_answer)} 字符")
                else:
                    print(f"[DEBUG] 没有找到 content")
                
                # 尝试获取原始响应的完整信息
                print(f"[DEBUG] 完整响应内容预览: {str(response)[:500]}...")
                
                # 检查是否有生成的图片
                generated_images = []
                image_generated = False
                
                # 检查模型是否支持图片生成
                is_image_generation_model = any(keyword in model_name.lower() for keyword in [
                    'dall-e', 'dalle', 'midjourney', 'stable-diffusion', 'image-gen'
                ])
                
                if is_image_generation_model:
                    # 对于图片生成模型，检查响应中的图片数据
                    if hasattr(response, 'data') and response.data:
                        print(f"[DEBUG] Found {len(response.data)} images in response")
                        for i, item in enumerate(response.data):
                            try:
                                if hasattr(item, 'b64_json') and item.b64_json:
                                    img_data = base64.b64decode(item.b64_json)
                                    pil_img = Image.open(BytesIO(img_data))
                                    generated_images.append(pil_img)
                                elif hasattr(item, 'url') and item.url:
                                    print(f"[DEBUG] Image URL: {item.url}")
                                    # 可以添加下载URL图片的逻辑
                            except Exception as img_e:
                                print(f"[DEBUG] Failed to process image {i}: {img_e}")
                    
                    image_generated = len(generated_images) > 0
                else:
                    # 对于视觉理解模型（如 Gemini），不期望生成图片
                    print(f"[INFO] 模型 {model_name} 是视觉理解模型，不生成新图片")
                    image_generated = False
                
                if generated_images:
                    out_tensor = _pil_list_to_comfy_images(generated_images)
                    image_generated = True
                else:
                    # 未生成图片则回传第一张输入图片
                    out_tensor = image1
                    image_generated = False
                
                # 如果没有获得任何内容，返回调试信息
                if not reasoning_content and not final_answer:
                    debug_info = f"[调试信息] API调用成功但未返回内容。模型: {model_name}, 响应选择数: {len(response.choices)}"
                    return out_tensor, debug_info, "API响应为空，请检查模型名称和API配置", image_generated
                
                return out_tensor, reasoning_content, final_answer, image_generated
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 对于某些错误，不进行重试
                if any(keyword in error_str for keyword in [
                    "400", "401", "403", "FAILED_PRECONDITION", 
                    "User location is not supported", "PROHIBITED_CONTENT"
                ]):
                    break
                
                # 对于可重试的错误，等待后重试
                if attempt < max_retries:
                    import time
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
        
        # 处理最终失败
        error_str = str(last_error) if last_error else "Unknown error"
        
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            error_msg = "[OpenAI Gemini Node] API配额已用完，请稍后重试或升级计划。"
        elif "401" in error_str:
            error_msg = "[OpenAI Gemini Node] API Key 无效，请检查环境变量设置。"
        elif "400" in error_str and "User location is not supported" in error_str:
            error_msg = "[OpenAI Gemini Node] 地理位置限制：你所在的地区不支持使用此 API。"
        elif "PROHIBITED_CONTENT" in error_str:
            error_msg = "[OpenAI Gemini Node] 内容被拒绝：模型认为请求包含禁止的内容。"
        else:
            error_msg = f"[OpenAI Gemini Node] API call failed: {last_error}"
        
        return image1, "", error_msg, False


class GoogleImagenGenerate:
    """
    ComfyUI Custom Node: Google Gemini 图片编辑生成节点
    
    功能:
    - 使用 Google 原生 API 调用 Gemini 模型进行图片编辑和生成
    - 支持多张图片输入：第1张必选，第2-5张可选
    - 支持基于输入图片进行编辑、风格转换、内容修改
    - 从环境变量获取 Google API Key
    - 支持多种图片尺寸和数量配置
    - 支持 seed 控制生成的随机性
    
    图片编辑能力:
    - 风格转换：将输入图片转换为不同艺术风格
    - 内容编辑：修改图片中的物体、背景、颜色等
    - 图片融合：结合多张输入图片创造新内容
    - 图片增强：提升图片质量、分辨率等
    
    支持的模型:
    - gemini-2.5-flash-image-preview: Gemini 2.5 图片预览编辑模型（默认）
    - imagen-3.0: Google Imagen 3.0 图片生成模型
    - 支持高质量图片生成和编辑
    - 支持中文提示词
    
    输入图片:
    - image1: 第一张图片（必选）- 主要编辑目标
    - image2: 第二张图片（可选）- 参考或融合素材
    - image3: 第三张图片（可选）- 参考或融合素材
    - image4: 第四张图片（可选）- 参考或融合素材
    - image5: 第五张图片（可选）- 参考或融合素材
    
    依赖:
    - pip install requests Pillow numpy torch
    - 需配置环境变量：DEEPSEEK_API_KEY、OPENAI_API_KEY 或 GOOGLE_API_KEY
    
    注意:
    - 当前配置使用 OpenAI 兼容 API 格式
    - 主要返回图片分析和编辑建议文本
    - 如需真正的图片生成，可能需要不同的API端点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "将这张图片转换为油画风格，保持主要内容不变，增强艺术感和色彩表现力"
                }),
                "image1": ("IMAGE", {}),  # 第一张图片必选
                "model": (["gemini-2.5-flash-image-preview", "imagen-3.0"], {
                    "default": "gemini-2.5-flash-image-preview"
                }),
                "size": (["1024x1024", "512x512", "256x256"], {
                    "default": "1024x1024"
                }),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "生成图片的数量 (1-4张)"
                }),
            },
            "optional": {
                "image2": ("IMAGE", {}),  # 第二张图片可选
                "image3": ("IMAGE", {}),  # 第三张图片可选
                "image4": ("IMAGE", {}),  # 第四张图片可选
                "image5": ("IMAGE", {}),  # 第五张图片可选
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "随机种子，-1为自动生成"
                }),
                "max_retries": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("images", "info", "success", "image_count")
    FUNCTION = "generate"
    CATEGORY = "Google/GenAI"
    OUTPUT_NODE = False

    def generate(self, prompt, image1, model="gemini-2.5-flash-image-preview", size="1024x1024", n=1, 
                 image2=None, image3=None, image4=None, image5=None, seed=-1, max_retries=2):
        
        # 处理随机种子
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        # 获取 API Key - 尝试多个环境变量
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            return image1, "[Google Imagen Node] 未找到API Key，请设置 DEEPSEEK_API_KEY、OPENAI_API_KEY 或 GOOGLE_API_KEY 环境变量", False, 0
        
        # 处理输入图片
        input_images = []
        try:
            # 处理第一张图片（必选）
            pil_img1 = _comfy_image_to_pil(image1)
            base64_img1 = _pil_to_base64(pil_img1)
            input_images.append(base64_img1)
            
            # 处理其他图片（可选）
            for i, img in enumerate([image2, image3, image4, image5], 2):
                if img is not None:
                    pil_img = _comfy_image_to_pil(img)
                    base64_img = _pil_to_base64(pil_img)
                    input_images.append(base64_img)
                    print(f"[DEBUG] 添加第{i}张输入图片")
            
            print(f"[DEBUG] 总共处理了 {len(input_images)} 张输入图片")
            
        except Exception as e:
            return image1, f"[Google Imagen Node] 图片处理失败: {e}", False, 0
        
        # 构建 API URL
        #url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateImages"
        url = f"https://www.chataiapi.com/v1/chat/completions"
        # 设置请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建OpenAI兼容格式的消息
        content_list = [{"type": "text", "text": str(prompt)}]
        
        # 添加输入图片到消息内容
        for i, img_base64 in enumerate(input_images):
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # 构建请求载荷（OpenAI格式）
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ]
        }
        
        # 如果需要seed控制（API支持的话）
        if seed > 0:
            payload["seed"] = seed
        
        print(f"[DEBUG] 调用 Google API - 模型: {model}")
        print(f"[DEBUG] 提示词: {prompt[:100]}...")
        print(f"[DEBUG] 参数: size={size}, n={n}, seed={seed}")
        
        # 特别提示：gemini-2.5-flash-image-preview 主要是视觉理解模型
        if "gemini" in model.lower() and "preview" in model.lower():
            print(f"[WARNING] 注意：{model} 主要是视觉理解模型，图片生成能力可能有限")
        
        # API 调用重试逻辑
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                print(f"[DEBUG] API响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"[DEBUG] 响应数据结构: {list(data.keys())}")
                    
                    # 解析OpenAI格式的响应
                    generated_images = []
                    response_text = ""
                    
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        message = choice.get("message", {})
                        
                        # 获取文本响应
                        if "content" in message:
                            response_text = message["content"]
                            print(f"[DEBUG] 模型响应文本: {response_text[:200]}...")
                        
                        # 检查是否有reasoning_content
                        if "reasoning_content" in message and message["reasoning_content"]:
                            print(f"[DEBUG] 找到推理内容: {message['reasoning_content'][:100]}...")
                        
                        # 注意：OpenAI聊天API通常不直接返回图片，主要是文本响应
                        # 如果这是图片编辑API，可能需要不同的端点和响应格式
                        print(f"[INFO] 这是文本响应API，主要返回分析结果而不是生成图片")
                        
                        # 返回分析结果，使用原图片
                        return image1, f"📝 图片分析完成: {response_text}", True, 1
                    
                    else:
                        print(f"[DEBUG] 响应中没有找到 'choices' 字段")
                        print(f"[DEBUG] 完整响应: {data}")
                        return image1, "[Google Imagen Node] API 返回格式异常", False, 0
                
                else:
                    error_data = response.text
                    print(f"[DEBUG] API错误响应: {error_data}")
                    
                    try:
                        error_json = response.json()
                        error_msg = error_json.get("error", {}).get("message", "未知错误")
                    except:
                        error_msg = f"HTTP {response.status_code}: {error_data[:200]}"
                    
                    last_error = Exception(f"API错误: {error_msg}")
                    
                    # 对于某些错误不重试
                    if response.status_code in [400, 401, 403]:
                        break
                        
            except requests.exceptions.Timeout:
                last_error = Exception("请求超时")
            except Exception as e:
                last_error = e
                print(f"[DEBUG] 请求异常: {e}")
            
            # 等待后重试
            if attempt < max_retries:
                import time
                wait_time = (attempt + 1) * 2
                print(f"[DEBUG] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        # 最终失败处理
        error_str = str(last_error) if last_error else "未知错误"
        
        if "401" in error_str or "authentication" in error_str.lower():
            error_msg = "[Google Imagen Node] API Key 无效或未授权，请检查 GOOGLE_API_KEY 环境变量"
        elif "quota" in error_str.lower() or "429" in error_str:
            error_msg = "[Google Imagen Node] API 配额已用完，请稍后重试"
        elif "timeout" in error_str.lower():
            error_msg = "[Google Imagen Node] 请求超时，请稍后重试"
        else:
            error_msg = f"[Google Imagen Node] 图片编辑失败: {error_str}"
        
        return image1, error_msg, False, 0


NODE_CLASS_MAPPINGS = {
    "GeminiGenerate": GeminiGenerate,
    "OpenAIGeminiGenerate": OpenAIGeminiGenerate,
    "GoogleImagenGenerate": GoogleImagenGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiGenerate": "Gemini Generate",
    "OpenAIGeminiGenerate": "OpenAI Gemini Generate", 
    "GoogleImagenGenerate": "Google Gemini Generate",
}

