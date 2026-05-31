---
title: 搜索系统模块架构说明
tags: [模块化, 架构, ingest, index, search, app]
---
# 搜索系统模块架构说明

## ingest 模块

ingest 模块负责扫描目录、读取 Markdown 和 TXT 文件，并解析标题、标签、小标题、正文
和修改时间。

## index 模块

index 模块负责构建倒排索引，并把解析后的文档保存到本地缓存。倒排索引是搜索系统的
核心数据结构。

## search 模块

search 模块负责查询解析、相关度评分、结果排序、上下文片段和关键词高亮。

## app 模块

app 模块负责命令行参数解析、普通文本输出、JSON 输出和 TUI 交互界面。
