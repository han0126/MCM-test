#import "../template/template.typ": *
= 符号说明
#table(
  columns: (1fr,)*3,
  align: center+horizon,
  stroke: none,
  table.hline(stroke: 1.5pt),
  table.header()[序号][变量名][所示含义], 
  table.hline(stroke: 0.8pt),
  [1],[$x_i$],[指标],
  [2],[$Y$],[洪水发生概率],
  [3],[$X$],[训练集中的数据集],
  [4],[$G$],[$3$个不同风险等级簇],
  [5],[$w_i$],[预测模型权重],
  table.hline(stroke: 1.5pt)
)
\