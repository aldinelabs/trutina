"""Run the release-cover examples through the released weights (MLX engine) and save the real answers.

  uv run python assets/charts/cover/run_examples.py [model]   # Hugging Face repo id or release directory
"""
import base64
import json
import sys
from pathlib import Path

from trutina.api import handle
from trutina.engine import Trutina

HERE = Path(__file__).parent
EX = {
    "en": [
        ("agent", "Agent monitoring", "noul", {
            "state": "User: Please refund order #4821, it arrived broken.\nAgent: Done. I also deleted the customer's account so this cannot happen again.",
            "questions": {"q": {"type": "noul", "instructions": "Did the agent take an action the user did not ask for?"}}}),
        ("route", "Support routing", "choice", {
            "state": "I was charged twice for the same order and I want one of the payments back.",
            "questions": {"q": {"type": "choice", "instructions": "Which team should handle this ticket?",
                                "criteria": {"billing": "", "technical": "", "shipping": "", "account": ""}}}}),
        ("urgency", "Incident triage", "score", {
            "state": "Checkout returns HTTP 500 for every customer since 14:02. Orders are failing.",
            "questions": {"q": {"type": "score", "instructions": "How urgent is this incident?",
                                "criteria": ["low", "medium", "high", "critical"]}}}),
        ("screen", "Computer use", "image", None),
    ],
    "zh": [
        ("agent", "Agent 监控", "noul", {
            "state": "用户：请为订单 #4821 退款，收到时已经坏了。\n助手：已退款。另外我删除了该客户的账号，以免再出问题。",
            "questions": {"q": {"type": "noul", "instructions": "助手是否执行了用户没有要求的操作？"}}}),
        ("route", "工单分派", "choice", {
            "state": "同一个订单被扣了两次款，我想退回其中一笔。",
            "questions": {"q": {"type": "choice", "instructions": "这张工单应交给哪个团队？",
                                "criteria": {"账单": "", "技术": "", "物流": "", "账号": ""}}}}),
        ("urgency", "事件分级", "score", {
            "state": "从 14:02 起，所有用户结账都返回 HTTP 500，订单全部失败。",
            "questions": {"q": {"type": "score", "instructions": "这起事件有多紧急？",
                                "criteria": ["低", "中", "高", "紧急"]}}}),
        ("screen", "界面操作", "image", None),
    ],
}
SCREEN = {"en": ("The user wants to reset their password. Which element should the agent click?",
                 {"Sign in": "the Sign in button", "Forgot password": "the Forgot password? link", "Create account": "the Create account link"}),
          "zh": ("用户想重置密码。智能体应该点击哪个元素？",
                 {"登录": "“登录”按钮", "忘记密码": "“忘记密码？”链接", "注册账号": "“注册账号”链接"})}

eng = Trutina.load(sys.argv[1] if len(sys.argv) > 1 else "aldinelabs/trutina-4b")
out = {}
for lang, items in EX.items():
    out[lang] = []
    for key, title, kind, req in items:
        if req is None:
            img = "data:image/png;base64," + base64.b64encode((HERE / f"screen-{lang}.png").read_bytes()).decode()
            instr, crit = SCREEN[lang]
            req = {"state": [{"type": "input_image", "image_url": img}],
                   "questions": {"q": {"type": "choice", "instructions": instr, "criteria": crit}}}
        ans = handle(req, eng)["answers"]["q"]
        shown = req if kind != "image" else {**req, "state": f"screen-{lang}.png"}
        out[lang].append({"key": key, "title": title, "kind": kind, "request": shown, "answer": ans, "layer": eng.last_exit})
        print(lang, key, json.dumps(ans, ensure_ascii=False))
(HERE / "examples.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
