#!/bin/sh
# 编译脚本：xelatex 两遍
cd "$(dirname "$0")"
mkdir -p build
for i in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build 对偶理论与灵敏度分析_基础学习讲义.tex > build/xelatex.out 2>&1 || { grep -E "^!" -A5 build/xelatex.out | head -30; exit 1; }
done
grep -c "Overfull" build/对偶理论与灵敏度分析_基础学习讲义.log
pdfinfo build/对偶理论与灵敏度分析_基础学习讲义.pdf | grep Pages
