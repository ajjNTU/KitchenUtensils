# main.py

from dataclasses import dataclass

@dataclass
class BotReply:
    text: str
    end_conversation: bool = False

def aiml_reply(user_input: str) -> 'BotReply | None':
    # TODO: Implement AIML reply
    return None

def tfidf_reply(user_input: str) -> 'BotReply | None':
    # TODO: Implement TF-IDF reply
    return None

def embed_reply(user_input: str) -> 'BotReply | None':
    # TODO: Implement embedding reply
    return None

def logic_reply(user_input: str) -> 'BotReply | None':
    # TODO: Implement logic reply
    return None

def vision_reply(user_input: str) -> 'BotReply | None':
    # TODO: Implement vision reply
    return None

def main():
    print("Welcome to the Kitchen Utensils Chatbot (Prototype)")
    while True:
        user_input = input("> ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        # Routing order: AIML → TF-IDF → Embedding → fallback
        reply = (
            aiml_reply(user_input)
            or tfidf_reply(user_input)
            or embed_reply(user_input)
        )
        if reply is None:
            print("Sorry, I don't know that.")
            continue
        print(reply.text)
        if reply.end_conversation:
            print("Goodbye!")
            break

if __name__ == "__main__":
    main() 