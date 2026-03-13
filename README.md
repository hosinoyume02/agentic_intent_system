# Agentic Intent System

一个基于 **BERT 的意图识别（Intent Classification）+ 槽位抽取（Slot
Filling）+ 多 Agent 调度架构** 的智能对话系统示例项目。\
系统模拟
**智能车载助手场景**，能够理解用户的自然语言指令，并自动分发给不同业务
Agent 进行处理，例如车辆控制、导航定位、售后服务和故障救援等。

------------------------------------------------------------------------

# 项目功能

本项目实现了一个完整的 **自然语言理解（NLU）+ Agent
执行框架**，主要包含以下功能：

-   用户自然语言输入解析
-   BERT 意图识别
-   BERT 槽位抽取
-   AgentManager 任务调度
-   多业务 Agent 执行
-   上下文管理（多轮对话支持）

系统处理流程如下：

    用户输入
       ↓
    意图识别 (Intent Classification)
       ↓
    槽位抽取 (Slot Filling)
       ↓
    AgentManager 调度
       ↓
    业务 Agent 处理
       ↓
    返回执行结果

------------------------------------------------------------------------

# 项目结构

    agentic_intent_system
    │
    ├── agent_demo.py            # 系统演示入口
    │
    ├── agentic
    │   ├── agent_manager.py     # Agent 调度管理器
    │   ├── context_manager.py   # 对话上下文管理
    │   │
    │   └── agents               # 各业务 Agent
    │       ├── intent_classifier.py
    │       ├── vehicle_control_agent.py
    │       ├── navigation_agent.py
    │       ├── service_booking_agent.py
    │       └── fault_assist_agent.py
    │
    ├── data                     # 训练数据
    │   ├── intent_train.json
    │   └── slot_train.json
    │
    ├── train.py                 # 意图识别模型训练
    ├── train_slot.py            # 槽位抽取模型训练
    ├── infer_slot.py            # 槽位抽取推理
    │
    └── README.md

------------------------------------------------------------------------

# 技术栈

本项目主要使用以下技术：

-   Python\
-   PyTorch\
-   HuggingFace Transformers\
-   BERT 预训练模型\
-   Intent Classification（意图识别）\
-   Slot Filling（槽位抽取）\
-   Agent Architecture（多 Agent 架构）

------------------------------------------------------------------------

# 核心模块说明

## 1. 意图识别（Intent Classification）

使用 **BERT Sequence Classification** 对用户输入进行意图分类。

示例：

输入：

    打开空调

输出：

    车辆控制-开空调

模型实现：

    BertForSequenceClassification

------------------------------------------------------------------------

## 2. 槽位抽取（Slot Filling）

使用 **BERT Token Classification**
进行序列标注，提取用户指令中的关键参数。

示例：

输入：

    把空调调到22度冷风

抽取结果：

    temperature: 22
    mode: 冷风

采用 **BIO 标注方式**进行实体识别。

------------------------------------------------------------------------

## 3. Agent 调度系统

系统通过 `AgentManager` 根据识别的意图自动选择对应业务 Agent。

示例 Agent：

-   车辆控制 Agent
-   导航定位 Agent
-   售后服务 Agent
-   故障救援 Agent

核心逻辑：

    intent = "车辆控制-开空调"
    main_intent = intent.split("-")[0]

然后调用对应 Agent：

    agent.handle()

------------------------------------------------------------------------

## 4. 上下文管理

通过 `ContextManager`
维护用户的对话上下文，实现简单的多轮对话支持，例如：

-   记录用户历史指令
-   维护当前对话状态
-   支持参数补全

------------------------------------------------------------------------

# 数据集说明

项目包含两个训练数据集：

## 意图分类数据

    data/intent_train.json

示例：

    {
      "text": "打开空调",
      "intent": "车辆控制-开空调"
    }

------------------------------------------------------------------------

## 槽位抽取数据

    data/slot_train.json

采用 BIO 标注方式：

    把 空调 调 到 22 度 冷 风
    O O O O B-temp I-temp O B-mode I-mode

------------------------------------------------------------------------

# 模型训练

## 训练意图识别模型

    python train.py

------------------------------------------------------------------------

## 训练槽位抽取模型

    python train_slot.py

------------------------------------------------------------------------

# 推理测试

运行系统 Demo：

    python agent_demo.py

示例输入：

    导航到北京大学

系统输出：

    Intent: 导航定位
    Slots: {location: 北京大学}
    Agent: NavigationAgent
    Response: 开始导航

------------------------------------------------------------------------

# 项目特点

-   基于 **BERT 的意图识别与槽位抽取系统**
-   实现 **NLU + Agent 执行框架**
-   支持 **多业务 Agent 调度**
-   模拟 **智能车载助手场景**
-   提供完整 **训练 + 推理代码**

------------------------------------------------------------------------

# 未来改进方向

未来可以进一步扩展：

-   引入 **LLM 进行意图理解**
-   支持 **更复杂的多轮对话管理**
-   增加更多业务 Agent
-   构建 Web 或 API 服务接口
-   支持实时语音输入

------------------------------------------------------------------------

