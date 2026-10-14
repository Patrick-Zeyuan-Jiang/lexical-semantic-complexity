# ODE WSD — frozen original source

本仓库保留现有 WSD 的最终 Python 源码，包括原 notebook 的义项编号冲突隔离补丁。
`ode_wsd/` 内源码按原字节保存。安装配置、说明和校验工具位于包目录之外。

## 安装与使用

仓库公开后，在 Colab 中按实际仓库地址和完整提交编号安装：

```python
%pip install "git+https://github.com/OWNER/REPOSITORY.git@FULL_COMMIT_SHA"
```

首次在新环境中使用时，准备原工具要求的 NLTK 资源：

```python
from ode_wsd.nlp import setup_nltk
setup_nltk()
```

然后提供与源码匹配的已有 bank 目录：

```python
from ode_wsd.wsd import WSDTool
tool = WSDTool(BANK_DIR, pos_filter="none")
rows = tool.disambiguate("She sat on the bank of the river.", include_uncovered=True)
```

`fine_sense_id` 是余弦比较的直接赢家；`main_sense_id` 是归并后的主义项。
源码会检查已有 bank 的源码指纹及文件哈希。默认使用库 manifest 锁定的编码配置。

## 资源与验证范围

本仓库不包含 ODE 原始词典、例句数据、bank、模型权重或 BNC 语料。
只有本仓库代码不足以直接运行消歧；使用者还需要取得匹配的 bank。
词典及其衍生资源的公开方式与源码许可分别确定。

`EXPORT_VERIFICATION.json` 记录导出时对已有 bank 的字节/指纹校验结果。
该检查没有运行模型，也没有重新评测准确率；它不锁定全部依赖或 NLTK 资源。
`requirements.txt` 保留原 notebook 的依赖声明，其中部分范围并非精确环境锁。

下载仓库后可以用标准库核验冻结源码：

```bash
python tools/verify_frozen_source.py
```

原工具的主次义项规则、词形匹配、子词平均、第7层编码和余弦选择由原源码定义。
新指标/语料适配代码应放在 `ode_wsd/` 之外。

## 许可与引用

作者需另行确定并添加适用的源码许可证；本次导出不自动授予数据资源的使用权限。
方法参考与项目来源：

- 原项目 notebook：见 `FROZEN_SOURCE.json` 的来源 SHA256。
- Teglia, Tedeschi & Navigli (2025): https://aclanthology.org/2025.acl-long.113/
- Lu & Hu: https://doi.org/10.3758/s13428-021-01675-6

仓库没有打包上述论文的原仓库、原始词典或预训练模型权重。
