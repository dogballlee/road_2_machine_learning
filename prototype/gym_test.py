import gym

env = gym.make('CartPole-v0')   # 生成cartpole-v0环境
state = env.reset()     # 重置环境，让小车回到起点，并初始化状态

for t in range(100):
    env.render()    # 弹出窗口，把游戏中发生的显示到屏幕上
    print(state)

    action = env.action_space.sample()  # 抽样生成一个动作，测试用。实际会用策略函数生成动作

    state, reward, done, info = env.step(action)    # 智能体执行动作，更新环境状态，反馈一个奖励

    if done:    # done的值1意味着游戏结束，0意味着游戏继续
        print('Finished')
        break

env.close()
