---
title: 测试与代码质量检查
tags: [测试, cargo, clippy, fmt, 工程质量]
---
# 测试与代码质量检查

## 单元测试

单元测试覆盖分词、Markdown 解析、倒排索引、搜索排序、高亮和缓存加载。

## 集成测试

集成测试会真正运行命令行程序，验证 index、search 和 JSON 输出能够正常工作。

## 工程检查

提交前应运行 cargo fmt、cargo test 和 cargo clippy。通过这些检查可以说明项目具有基本
工程质量。
