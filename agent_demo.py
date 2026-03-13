from agentic.agent_manager import AgentManager

AM = AgentManager(intent_model_path="./outputs/")

while True:
    user_text = input("用户: ")
    if user_text.lower() in ("q", "quit", "exit"):
        break
    result = AM.dispatch(user="user001", text=user_text)
    print("助手:", result)