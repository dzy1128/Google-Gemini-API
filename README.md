# Google-Gemini-API
## 节点1：Gemini Generate
使用Google官方的API，目前价格较高

## 节点2：Gemini Edit Image(推荐使用)
使用购买的供应商的API，目前测试的价格约为0.08-0.1元每张

## 节点2参数说明：
seed: 随机种子
num_outputs：出图数量
model_name：模型名称，目前默认使用gemini-2.5-flash-image-preview
api_url：API接口地址
max_retries：生成失败自动重试的次数，如果想节约API填0即可。