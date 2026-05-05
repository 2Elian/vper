"""
P2P Mailbox - Agent 间消息传递

支持 Agent 间点对点通信和广播
"""


class Message(object):
    """消息"""
    def __init__(self, msg_type, from_agent, to_agent, payload):
        # type: (str, str, str, dict) -> None
        self.msg_type = msg_type  # request_data, share_result, notify, handoff
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.payload = payload

    def to_dict(self):
        # type: () -> dict
        return {
            "type": self.msg_type,
            "from": self.from_agent,
            "to": self.to_agent,
            "payload": self.payload,
        }


class Mailbox(object):
    """Agent 间点对点消息系统"""

    def __init__(self):
        self.boxes = {}  # type: dict  # agent_id -> [Message]

    def send(self, message):
        # type: (Message) -> None
        target = message.to_agent
        if target not in self.boxes:
            self.boxes[target] = []
        self.boxes[target].append(message)

    def receive(self, agent_id):
        # type: (str) -> list
        messages = self.boxes.pop(agent_id, [])
        return messages

    def broadcast(self, from_agent, msg_type, payload):
        # type: (str, str, dict) -> None
        for agent_id in self.boxes:
            if agent_id != from_agent:
                self.boxes[agent_id].append(
                    Message(msg_type, from_agent, agent_id, payload)
                )

    def has_messages(self, agent_id):
        # type: (str) -> bool
        return bool(self.boxes.get(agent_id))
