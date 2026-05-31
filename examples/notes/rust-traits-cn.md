---
title: Rust Trait 与泛型设计
tags: [rust, trait, 泛型, tokenizer]
---
# Rust Trait 与泛型设计

## Trait

Trait 可以定义一组行为。本项目中 Tokenizer trait 描述了分词器需要提供的 tokenize 方法。

## 泛型

倒排索引在添加文本时可以接收实现 Tokenizer trait 的分词器。这样代码结构更灵活，也能
体现 Rust 的抽象能力。

## 项目价值

使用 trait 和泛型不是为了炫技，而是为了让模块边界更清楚，让分词逻辑和索引逻辑解耦。
