### 影视解说-画面匹配（索引化版本）

#### 总体结构
- S1：视频集数索引化（按集数处理，不再合并）
- S2：结构化信息索引化（按集数分组）
- S3：解说文案拆分（融合suoying，按集数边界分段）
- S4：核心匹配与精排（基于episode_target在对应集内匹配）
- S5：视频合成（基于全局时间戳合成）
- S6：生成final.json

---

#### 用法
```bash
python run.py <dj> [start] [end] [output_filename] [top_k]
# 示例: python run.py buxi5 1 15
```

---

##### S1：视频集数索引化模块
###### 模块
`skill_video_episode_indexer.py`

###### SOP
- 选定视频集数范围（start ~ end）
- 按集数索引化，每集独立处理
- 调用 mllm 进行视频结构化理解
- 输出格式：`cache/<dj>/structured_description.json`
- 包含：视频片段id | 视频时长 | 视频起始时间 | 视频结束时间 | 结构化描述内容 | 片段文件路径 | 所属集数

---

##### S2：结构化信息索引化
###### 模块
`skill_struct_info_emb.py`

###### SOP
- 请求 bgm/bert 获取语义向量
- 按集数分组保存索引
- 输出：`cache/<dj>/embeddings.pkl`

---

##### S3：解说文案拆分（集数索引版）
###### 模块
`skill_text_structured_description.py`

###### SOP
- 融合 suoying 信息，按集数边界分段
- 语义切割：调用大模型将文案切分为解说单元
- 搜索词生成：把解说内容转换成搜索词
- 解说单元请求 TTS，音频结果保存，记录音频时长
- 输出：`cache/<dj>/<txt_name>_structured.json`
- 包含：文案内容 | 音频文件路径 | 音频时长 | episode_target

---

##### S4：核心匹配与精排（集数索引版）
###### 模块
`skill_match_and_rank.py`

###### SOP
- 读取每个解说单元，根据 episode_target 在对应集内匹配
- 向量化找到视频描述 topk（要求视频时长与解说时长近似）
- 利用 LLM 选择最终候选片段
- 结果记录：解说单元 id --> 视频片段 id
- 更新至 S3 的 JSON 文件

---

##### S5：视频合成模块
###### 模块
`skill_postprocess.py`

###### SOP
- 读取视频结构化信息（视频id，视频时长，视频文件路径）
- 读取解说单元结构化信息（解说单元id，解说音频文件等）
- 读取解说单元 id --> 视频片段 id 映射
- 根据视频时长调整语速，与解说时长一致
- 片段合成，输出最终视频
- 输出：`finish/<output_filename>`

---

##### S6：生成final.json
###### 说明
- 基于 structured JSON，只保留每个 segment 的 top1 匹配
- 输出：`cache/<dj>/<dj>_final.json`