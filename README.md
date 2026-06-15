# RustNoteSearch

RustNoteSearch 是一个基于 Rust 实现的本地 Markdown/TXT 知识库搜索系统。

项目可以扫描用户指定目录下的 Markdown 和 TXT 文件，解析文件路径、文件名、标题、
小标题、正文、标签、修改时间等信息，构建本地倒排索引并保存缓存。搜索时支持一个
或多个关键词查询，并返回按相关度排序的结果，包含匹配文件、命中章节、相关度分数、
关键词附近的上下文片段和高亮效果。

本项目没有调用大型搜索引擎库。文件扫描、文档解析、分词、倒排索引、相关度排序、
JSONL 缓存序列化、文本高亮和 TUI 交互界面均由项目代码实现。

## 功能特点

- 递归扫描 `.md`、`.markdown`、`.txt` 文件。
- 解析文件路径、文件名、标题、小标题、正文、标签和修改时间。
- 支持 Markdown 标题、front matter 中的 `title` / `tags`、正文中的 `#标签`、
  TXT 文件中的 `Tags:` 行。
- 自动忽略代码块中的伪 Markdown 标题，避免误解析。
- 构建本地倒排索引。
- 使用 JSON Lines 保存和加载索引缓存。
- 加载缓存时进行版本和记录类型校验。
- 支持一个或多个关键词搜索。
- 支持中文短语搜索和高亮。
- 相关度排序综合考虑标题命中、标签命中、小标题命中、正文命中、文件名命中、
  词频、IDF、多关键词覆盖和短语命中。
- 输出普通文本或 JSON 格式搜索结果。
- 提供轻量 TUI 终端交互界面，方便演示和连续搜索。

## 项目结构

```text
.
├── Cargo.toml
├── Cargo.lock
├── README.md
├── examples/
│   └── notes/                 示例中文知识库
├── src/
│   ├── main.rs                程序入口
│   ├── lib.rs                 库模块出口
│   ├── error.rs               统一错误类型
│   ├── model.rs               Document、Section、Field 等核心模型
│   ├── app/                   CLI、输出和 TUI 交互
│   ├── ingest/                文件扫描、Markdown/TXT 解析、分词
│   ├── index/                 倒排索引和缓存读写
│   └── search/                查询、排序、片段生成和高亮
└── tests/
    └── search_flow.rs         集成测试
```

核心模块说明：

- `ingest`：负责扫描目录、读取文件、解析 Markdown/TXT 元数据、分词。
- `index`：负责构建倒排索引，并保存/加载 JSONL 缓存。
- `search`：负责查询处理、相关度评分、结果排序、命中章节选择、上下文片段和高亮。
- `app`：负责命令行参数解析、普通文本输出、JSON 输出和 TUI 交互界面。

## 示例文档

项目自带中文示例知识库，位于：

```text
examples/notes
```

当前示例文档包括：

```text
01_overview.md                  机器学习概述
02_linear_models.md             线性模型
03_neural_networks.md           神经网络
04_deep_learning.md             深度学习
05_support_vector_machine.md    支持向量机
06_decision_tree.md             决策树
07_bayesian_classifiers.md      贝叶斯分类器
08_ensemble_learning.md         集成学习
09_clustering.md                聚类
10_dimensionality_reduction.md  降维与度量学习
11_model_evaluation.md          模型评估与调参
```

这些示例文档用于展示中文搜索、章节命中、相关度排序、上下文片段和 TUI 交互效果。

## 如何运行

进入项目目录：

```bash
cd D:\Desktop\大学\大三下\Rust\大作业
```

查看帮助：

```bash
cargo run -- help
```

构建示例知识库索引：

```bash
cargo run -- index examples\notes --cache target\demo-cache.jsonl
```

从缓存中搜索：

```bash
cargo run -- search 支持向量机 核函数 --cache target\demo-cache.jsonl --limit 5
```

扫描目录并立即搜索，同时刷新缓存：

```bash
cargo run -- search PCA 降维 --dir examples\notes --cache target\demo-cache.jsonl --limit 5
```

输出 JSON 搜索结果：

```bash
cargo run -- search 模型评估 交叉验证 --cache target\demo-cache.jsonl --format json
```

启动 TUI 交互界面：

```bash
cargo run -- tui examples\notes --cache target\tui-cache.jsonl --limit 5
```

TUI 中可以直接输入关键词并回车，例如：

```text
支持向量机 核函数
Logistic 回归
神经网络 反向传播
PCA 降维
模型评估 交叉验证
```

TUI 支持的内部命令：

```text
:help       查看帮助
:rebuild    重新扫描目录并刷新缓存
:clear      重绘界面
:quit       退出
```

## 命令说明

```text
md-knowsearch index <directory> [--cache <file>]
md-knowsearch rebuild <directory> [--cache <file>]
md-knowsearch search <keywords...> [--cache <file>] [--dir <directory>] [--rebuild] [--limit <n>] [--format text|json]
md-knowsearch tui <directory> [--cache <file>] [--limit <n>]
```

说明：

- `index`：扫描目录并构建缓存。
- `rebuild`：重新扫描目录并覆盖缓存。
- `search`：搜索一个或多个关键词。
- `tui`：启动终端交互式搜索界面。
- `--cache`：指定缓存文件路径。
- `--dir`：搜索前先扫描指定目录并刷新缓存。
- `--limit`：限制返回结果数量。
- `--format json`：使用 JSON 输出搜索结果。

## 推荐演示流程

1. 展示项目简介：本地 Markdown/TXT 知识库搜索系统。
2. 展示 `examples/notes` 中的中文示例文档。
3. 运行工程检查：

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

4. 构建索引：

```bash
cargo run -- index examples\notes --cache target\demo-cache.jsonl
```

5. 展示普通搜索：

```bash
cargo run -- search 支持向量机 核函数 --cache target\demo-cache.jsonl --limit 3
```

6. 展示 JSON 输出：

```bash
cargo run -- search 模型评估 交叉验证 --cache target\demo-cache.jsonl --format json
```

7. 启动 TUI，并连续输入几个查询：

```bash
cargo run -- tui examples\notes --cache target\tui-cache.jsonl --limit 5
```

推荐 TUI 查询：

```text
支持向量机 核函数
Logistic 回归
神经网络 反向传播
PCA 降维
模型评估 交叉验证
```

演示重点：

- 系统可以扫描本地知识库。
- 能解析标题、标签、小标题、正文和修改时间。
- 能构建倒排索引并保存缓存。
- 搜索结果有相关度分数、命中章节、标签、关键词高亮和上下文片段。
- 支持普通文本、JSON 和 TUI 三种使用方式。

## 相关度排序设计

搜索结果不是简单字符串匹配。系统对每个查询词计算综合分数，主要考虑：

- 标题命中权重较高。
- 标签命中权重较高。
- 小标题命中高于普通正文命中。
- 正文中重复出现的词会增加词频分数。
- 多个关键词同时命中会获得覆盖奖励。
- 多词短语直接出现会获得额外奖励。
- 文件名和修改时间也会参与少量评分。

这样可以让更符合用户意图的文档排在前面。

## 工程实践

本项目体现了以下 Rust 工程实践：

- 使用模块化结构组织代码。
- 使用 `struct` 表示文档、章节、搜索结果等数据。
- 使用 `enum` 表示字段类型、命令类型和错误类型。
- 使用 `trait` 抽象分词器行为。
- 使用 `Result` 进行错误处理和错误传播。
- 避免大量使用 `unwrap` / `expect`。
- 使用单元测试和集成测试验证关键功能。
- 提交前通过 `cargo fmt`、`cargo test` 和 `cargo clippy`。

## 测试与检查

运行格式检查：

```bash
cargo fmt --check
```

运行测试：

```bash
cargo test
```

运行 Clippy：

```bash
cargo clippy -- -D warnings
```

当前测试覆盖内容包括：

- 分词与中文短语分词。
- Markdown/TXT 元数据解析。
- 代码块标题保护。
- 文件扫描错误处理。
- 倒排索引构建。
- 缓存保存、加载和格式校验。
- 搜索排序。
- 上下文片段和高亮。
- TUI 命令解析。
- CLI 端到端流程。

## 提交说明

提交源码时不要包含 `target/` 目录。`target/` 是 Cargo 自动生成的构建输出目录，
体积较大，其他人运行 `cargo build`、`cargo run` 或 `cargo test` 时会自动重新生成。

建议提交内容：

```text
Cargo.toml
Cargo.lock
README.md
src/
tests/
examples/
.gitignore
```
