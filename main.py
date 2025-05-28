# main.py

def main():
    print("Welcome to the Kitchen Utensils Chatbot (Prototype)")
    # Future: Initialize modules and router here
    while True:
        user_input = input("> ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        # Future: Route to AIML/NLP/Logic/Image modules
        print("[Stub] Feature coming soon.")

if __name__ == "__main__":
    main() 