from dataclasses import dataclass
from inferenceagent import AgentCall



class SavedTrajectoryNode:
    def __init__(self):
        self.ad_call: AgentCall | None = None
        self.reflect_call: list[AgentCall] = []
        self.world_mem_call: AgentCall | None = None
        self.url: str | None = None
        self.screenshot: bytes | None = None


class SavedTrajectory:
    def __init__(self):
        self.nodes = []
        self.user_input_task = None
        self.question_answers = None
        self.cleaned_task = None
        self.important_info = None

    def add_node(self, node):
        self.nodes.append(node)

    def step_through(self):
        index = 0
        while (index < len(self.nodes)) and (index >= 0):
            node = self.nodes[index]
            print(f"\n--- Node {index} ---")
            print(f"URL: {node.url}")
            print("Options:")
            print("1) View AD call")
            print("2) View Reflect call")
            print("3) View World Mem call")
            print("4) Go to next node")
            print("5) Go to previous node")
            print("6) Quit")

            decision = input("Enter your choice (1-6): ").strip()

            if decision == '1':
                if node.ad_call:
                    print("\n--- AD Call ---")
                    # if node.ad_call.system_prompt:
                    #     print("System Prompt:")
                    #     print(node.ad_call.system_prompt)
                    # else:
                    #     print("No system prompt available.")
                    #
                    # if node.ad_call.user_prompt:
                    #     print("\nUser Prompt:")
                    #     print(node.ad_call.user_prompt)
                    # else:
                    #     print("No user prompt available.")

                    for message in node.ad_call.messages:
                        print(message)

                    if node.ad_call.llm_response:
                        print("\nLLM Response:")
                        print(node.ad_call.llm_response)
                    else:
                        print("No LLM response available.")
                else:
                    print("\nNo AD call available for this node.")

                input("\nPress Enter to continue...")

            elif decision == '2':
                if node.reflect_call:
                    print("\n--- Reflect Call ---")
                    for message in node.ad_call.messages:
                        print(message)
                    # if node.reflect_call.system_prompt:
                    #     print("System Prompt:")
                    #     print(node.reflect_call.system_prompt)
                    # else:
                    #     print("No system prompt available.")
                    #
                    # if node.reflect_call.user_prompt:
                    #     print("\nUser Prompt:")
                    #     print(node.reflect_call.user_prompt)
                    # else:
                    #     print("No user prompt available.")

                    if node.reflect_call.llm_response:
                        print("\nLLM Response:")
                        print(node.reflect_call.llm_response)
                    else:
                        print("No LLM response available.")
                else:
                    print("\nNo Reflect call available for this node.")

                input("\nPress Enter to continue...")

            elif decision == '3':
                if node.world_mem_call:
                    print("\n--- World Mem Call ---")
                    if node.world_mem_call.system_prompt:
                        print("System Prompt:")
                        print(node.world_mem_call.system_prompt)
                    else:
                        print("No system prompt available.")

                    if node.world_mem_call.user_prompt:
                        print("\nUser Prompt:")
                        print(node.world_mem_call.user_prompt)
                    else:
                        print("No user prompt available.")

                    if node.world_mem_call.llm_response:
                        print("\nLLM Response:")
                        print(node.world_mem_call.llm_response)
                    else:
                        print("No LLM response available.")
                else:
                    print("\nNo World Mem call available for this node.")

                input("\nPress Enter to continue...")

            elif decision == '4':
                if index + 1 < len(self.nodes):
                    index += 1
                    print(f"\nMoving to next node: {index}")
                else:
                    print("\nYou are at the last node. Cannot go to the next node.")
                input("\nPress Enter to continue...")

            elif decision == '5':
                if index - 1 >= 0:
                    index -= 1
                    print(f"\nMoving to previous node: {index}")
                else:
                    print("\nYou are at the first node. Cannot go to the previous node.")
                input("\nPress Enter to continue...")

            elif decision == '6':
                print("\nExiting step-through.")
                break

            else:
                print("\nInvalid choice. Please enter a number between 1 and 6.")
                input("\nPress Enter to continue...")
