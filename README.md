# pdf_attack
基于cleverhans，tensorflow库。
1.训练神经网络模型作为分类器，网络层神经元个数为（5000，1000，1000，1000，1000），激活函数为tanh.输出层为softmax,输出二维向量,有害为［1，0］。分类器对所有数据的识别准确率为95.5%。<br>
2.进行随机攻击。选择了10000条被分类为有害的数据，攻击方法如下：<br>
	a. 随机生成四个0～4999的整数值，作为将要改变的维度。进行改变，规则为：若对应维为0或1则取反，若为常数则随机改变，改变量不超过原数据。<br>
value+(uniform(-1,1)*value)<br>
	b.将分类器结果的第二维看作分类为无害的概率，重复a步骤100次，记录每次无害概率的改变量，将100次中无害概率改变量最大的那次对应的改变维度纪录下来。将改变后数据作为下一轮的起始数据。<br>
	c.重复a,b直到数据被分类为无害为止。<br>
3.通过步骤2，得到了10000条数据经过随机变化为无害的改变路径。经过观察，大多数路径中间的改变对分类的影响较小，影响最大的是路径里最后一次改变。所以，统计10000条路径中最后一次改变的4个维度的频率。<br>
4.选择频率最高的前十维，对每一维，测试其对识别率的影响程度。<br>
测试方法：改变10000条数据中的某一维，对比改变前后分类准确率的改变量。<br>
频率最高的十维：   （由高到低）   
特征名   识别率变化   
1.ArcInfo{FileSize}    0.594421    
2.ArcInfo{ImageSize}   0.5098947   
3.sec.rsrc{VSize}    0.40842104   
4.sec.text{RawSize}   0.38294736   
5.sec.data{VSize}   0.29178947   
6.sec.data{RawSize}   0.16999996   
7.sec.rdata{VSize}   0.20042104   
8.sec.rsrc{RawSize}   0.12589473   
9.sec.rdata{RawSize}   0.0998947   
10.sec.reloc{VSize}   0.065368414   
使用cleverhans库，通过检验梯度大小的方法进行验证。梯度更大，说明在对应维方向上下降的越快，即对应维对结果的影响更大。   
对10000条数据，求得其每一维的梯度，取绝对值并累加，得到一个长为5000的向量。向量中每一维代表了其累加梯度，可看做其对结果的影响程度。通过大小比较，发现梯度值高的维与之前频率较高的维高度重合。   



