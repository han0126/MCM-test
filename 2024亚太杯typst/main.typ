#import "template/template.typ": *
#import "template/template.typ": template as APMCM

#show: APMCM.with(

  abstract: [
    洪水是全球造成重大经济损失和人员伤亡的自然灾害之一。有效预测和减轻其影响，本文综合分析了多项指标数据，并构建了*预测模型*。基于季风强度、地形排水等因素，通过数据清洗和相关性分析筛选出关键指标。运用各种神经网络模型，构建洪水预警模型，洪水概率预测模型。在数据分析部分，我们采用Python 软件的Seaborn 库，制作热力图、柱状图、折线图等*可视化*工具展示不同指标之间的相关性和分布特征。
    
    针对问题一，基于`train.csv`数据集，分析20 个不同指标与洪水概率的相关性。通过*相关性分析*以及可视化，提出了一系列洪水提前预防的建议，包括加强降水量和河流流量的实时监测、土壤湿度监测与调节、气象预报与警报系统的优化、防洪基础设施建设以及公众教育和应急演练。这些措施将有助于提高洪水预警的准确性和及时性，减轻洪水灾害的影响。
    
    对于问题二，我们将`train.csv`中的洪水发生概率进行聚类分析，以识别高、中、低风险的洪水事件。首先，使用K-means 聚类算法将数据分为三类，接着用XGBoost 算法构建预警模型。最后，通过交叉验证的方法，进行模型的灵敏度分析，得出验证出的模型准确率为96.515%。。
    
    针对问题三，基于问题1 中的指标分析结果，我们从20 个指标中选取5 个关键指标，建立洪水概率的预测模型。本组采用MLP 神经网络模型来对选取的指标数据进行分析。通过利用测试集中的数据，*MLP 神经网络*使用反向传播算法来优化网络参数，以最小化预测输出与实际输出之间的误差。最终，我们验证模型的准确性，达到99.97%，证明其能够有效地预测洪水发生的概率，为实际防洪提供了科学依据。
    
    对于问题四，基于问题3 中建立的洪水发生概率预测模型，我们对`test.csv`中的所有事件进行了洪水概率预测，并将预测结果填入`submit.csv`。进一步绘制了74 万件事件的洪水发生概率的直方图和折线图，我们根据Kolmogorov-Smirnov 方法检验预测值洪水概率是否服从正态分布。

    本文的研究结果不仅揭示了影响洪水发生的主要因素，还提供了一种有效的洪水预测方法，对防灾减灾具有重要的指导意义。未来的工作将进一步优化模型，并结合实时监测数据，提高预测的实时性和精确度。
  ],

  title: "基于机器学习的洪水灾害预测模型",
  keywords: ("洪水灾害预测", "可视化", "相关性分析", "K-means聚类","MLP神经网络")
)

#include "chapter/chapter1.typ"
#include "chapter/chapter2.typ"
#include "chapter/chapter3.typ"
#include "chapter/chapter4.typ"
#include "chapter/chapter5.typ"
#include "chapter/chapter6.typ"
#include "chapter/chapter7.typ"
#include "chapter/chapter8.typ"

= 参考文献
#bibliography("refr.bib", title: none, style: "gb-7714-2015-numeric")\ 

#include "chapter/appendix.typ"