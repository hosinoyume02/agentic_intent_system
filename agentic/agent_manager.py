from agentic.agents.intent_classifier import IntentClassifier
from agentic.agents.vehicle_control import VehicleControlAgent
from agentic.agents.navigation import NavigationAgent
from agentic.agents.service_booking import ServiceBookingAgent
from agentic.agents.fault_assist import FaultAssistAgent
from agentic.context_manager import ContextManager

class AgentManager:
    def __init__(self, intent_model_path):
        self.intent_classifier = IntentClassifier(intent_model_path)
        self.agents = {
            "车辆控制": VehicleControlAgent(),
            "导航定位": NavigationAgent(),
            "售后服务": ServiceBookingAgent(),
            "故障救援": FaultAssistAgent(),
            #"推荐服务": recommend_service_agent(),
            # ... add more agents
        }
        self.context_mgr = ContextManager()

    def dispatch(self, user, text):
        # 1. 意图识别
        '''
        {
            "intent": "车辆控制-开空调",     # 预测的意图标签
            "confidence": 0.98,             # 置信度（概率），如有
            "slots": {                      # 槽位（如做了槽位抽取）
                "temperature": "22",
                "model": "冷风"
            }
        }
        '''
        res = self.intent_classifier.predict(text)
        print(res)
        intent = res['intent']
        confidence = res['confidence']
        slots = res.get('slots', {})
        main_intent_cls = intent.split('-')[0]  # e.g. "车辆控制"
        # 2. 上下文更新（多轮）
        context = self.context_mgr.update(user, text, intent, slots)
        # 3. 分派下游Agent
        agent = self.agents.get(main_intent_cls)
        if not agent:
            return "暂不支持该业务"
        reply = agent.handle(text, intent, slots, user_context=context)
        return reply