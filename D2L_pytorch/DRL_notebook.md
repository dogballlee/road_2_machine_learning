# 强化学习笔记





## **强化学习定义**(个人理解版)：

通过对**智能体agent**的观察(包括当前**状态state**、采取**动作action**等)进行决策，尽量使**奖励reward**及**回报return**最大化的一种学习方式



## TD算法--时间差分(Time Difference)算法：

藉由已观测到的信息预测未来状态并规划行动策略的一种算法(例子：北京驾车前往上海，在济南停车计算已行驶里程和时间，观察剩余里程并再次预测驾驶剩余时间。按此方法得到的预测时间结果会比在北京时直接凭空预测的时间更靠谱合理)

算法步骤--Q-learning：

​				根据当前状态的四元组
$$
(s_t,a_t,r_t,s_{t+1})
$$
​				计算DQN的预测值
$$
\hat{q_t}=Q(s_t,a_t;\omega)
$$
​				及TD目标、TD误差
$$
\hat{y_t}=r_t+\gamma\cdot\max_{a\in A} Q(s_t,a_t;\omega)
$$

$$
\delta_t=\hat{q_t}-\hat{y_t}
$$

​				TD算法更新DQN参数
$$
\omega \gets \omega -\alpha \cdot\delta _t\cdot \nabla _m Q(s_t,a_t;\omega)
$$
​				将DQN的训练过程分为：收集训练数据、更新参数$\alpha$，具体如下：

​				收集训练数据：以任意策略函数**$\pi$**让智能体与环境交互(这个$\pi$即行为策略——Behavior Policy)，

​				

​				$a_t = $

​				--------------------------------------------待完善，latex里的大括号怎么敲啊...................

TD算法是一大类算法，上面使用的是其中的一种——**Q-learning**。除此以外还有**SARSA** 。Q-learning学习的是学到最优动作价值函数$Q_\star$，SARSA则是学习动作价值函数$ Q_\pi$

