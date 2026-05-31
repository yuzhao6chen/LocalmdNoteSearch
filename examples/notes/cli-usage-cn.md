---
title: 命令行使用示例
tags: [CLI, 命令行, 使用方法, JSON]
---
# 命令行使用示例

## 构建索引

可以使用 index 命令扫描指定目录，并生成本地缓存。

```bash
cargo run -- index examples/notes --cache target/demo-cache.jsonl
```

## 搜索关键词

可以使用 search 命令查询一个或多个关键词，例如 搜索 排序。

```bash
cargo run -- search 搜索 排序 --cache target/demo-cache.jsonl
```

## JSON 输出

如果需要把搜索结果交给其他程序处理，可以使用 --format json 输出结构化结果。
