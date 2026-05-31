---
title: Rust 错误处理实践
tags:
  - rust
  - 错误处理
  - result
---
# Rust 错误处理实践

## Result

Rust 中可恢复错误通常使用 Result 表示。函数可以返回 Result<T, E>，调用方根据 Ok
或 Err 分支决定后续行为。

## 错误传播

问号运算符可以把错误向上传播，使代码保持简洁。相比大量 unwrap，Result 和错误传播
更适合正式项目。

## 项目中的应用

本搜索系统在文件读取、缓存加载、命令行解析和 JSON 解析中都使用 Result 处理错误。
