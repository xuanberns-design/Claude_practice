# C++ 第七章 指针 · 基础学习讲义

根据课堂手写笔记（§7.1–§7.8，笔记页 378–393）重新制作的本科初学者可视化讲义。

| 文件 | 说明 |
| --- | --- |
| `C++指针_基础学习讲义.pdf` | 最终讲义（A4，74 页） |
| `C++指针_基础学习讲义.tex` | 单文件完整源码（XeLaTeX 编译两次） |
| `src/` | 分章节源码（`main.tex` 为入口，内容与单文件版一致） |
| `code/` | 讲义中全部 C++ 示例及练习答案核对程序（均已用 g++ C++17 编译运行） |
| `tools/` | 构建辅助脚本（排版检查、逐页预览） |

## 编译

```bash
xelatex "C++指针_基础学习讲义.tex"
xelatex "C++指针_基础学习讲义.tex"   # 第二次生成目录
```

依赖字体：Noto Serif CJK、Noto Sans CJK（`.ttc`，使用索引 2 的简体中文字形）、DejaVu Sans、DejaVu Sans Mono。
Debian/Ubuntu：`apt install texlive-xetex texlive-lang-chinese texlive-latex-extra texlive-pictures texlive-science fonts-noto-cjk fonts-dejavu`。

## 内容结构

导读与前置知识 → 学科小史 → 从课本到现实 → 原始笔记识别与 6 处纠错 → 第一讲～第八讲（内存与地址、`&`/`*`、指针大小、空/野指针、const、指针与数组、指针与函数、冒泡排序）→ 知识地图 → 38 道正式练习（25 基础 + 10 提升 + 3 跨学科）→ 答案速查 → 逐题图解解析 → 易错总表 / 步骤总结 / 概念总对比 / 后续课程 / 参考资料。
