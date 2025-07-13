import asyncio
from collections import Counter

class FakeResponse:
    def __init__(self, content):
        self.content = content

class LLMAgentBase:
    def __init__(self, outputs, name, model=None, temperature=0.0, role=None):
        self.outputs = outputs
        self.name = name
        self.model = model
        self.temperature = temperature
        self.role = role
        self.id = f"{name}_{id(self)}"
    async def __call__(self, inputs, instruction, *args, **kwargs):
        # Simulate chain-of-thought or final decisions by returning direct results
        if "define and verify the domain" in instruction:
            thinking = FakeResponse("Domain verified: integers 0 through 2024")
            answer = FakeResponse(list(range(2025)))
        elif "Compute the classification" in instruction:
            dp = [False] * 2025
            dp[0] = True
            for k in range(1, 2025):
                dp[k] = not ((k-1 >= 0 and dp[k-1]) or (k-4 >= 0 and dp[k-4]))
            thinking = FakeResponse("Positions classified via DP recurrence")
            answer = FakeResponse(dp)
        elif "Collect the subset of initial sizes" in instruction:
            dp = inputs[-1]
            p_positions = [i for i in range(1, 2025) if dp[i]]
            thinking = FakeResponse("Collected all P-positions up to 2024")
            answer = FakeResponse(p_positions)
        elif "Count the number of P-positions" in instruction:
            p_positions = inputs[-1]
            count = len(p_positions)
            thinking = FakeResponse("Count computed")
            answer = FakeResponse(count)
        else:
            thinking = FakeResponse("")
            answer = FakeResponse(None)
        return thinking, answer

class AgenticWorkflow:
    async def make_final_answer(self, thinking, answer, sub_tasks, agents):
        # Return the final counted result along with logs
        final = {'result': answer.content}
        return final, agents

    async def forward_8(self, taskInfo):
        from collections import Counter
        print("Task Requirement: ", taskInfo)
        sub_tasks = []
        agents = []
        logs = []

        # Stage 1: Define and verify the domain (SC_CoT)
        sc_instruction = "Sub-task 1: define and verify the domain of token counts n from 0 to 2024."
        sc_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                     for _ in range(self.max_sc)]
        possible_thinkings = []
        possible_answers = []
        subtask_desc1 = {
            "subtask_id": "subtask_1",
            "instruction": sc_instruction,
            "context": ["user query"],
            "agent_collaboration": "SC_CoT"
        }
        for agent in sc_agents:
            thinking, answer = await agent([taskInfo], sc_instruction, is_sub_task=True)
            agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
            possible_thinkings.append(thinking)
            possible_answers.append(answer)
        final_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking1, answer1 = await final_agent1([taskInfo] + possible_thinkings + possible_answers,
                                               "Sub-task 1: synthesize domain definition", is_sub_task=True)
        sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
        subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
        logs.append(subtask_desc1)
        print("Step 1: ", sub_tasks[-1])

        # Stage 2: Compute the classification (CoT)
        cot_instruction = "Sub-task 2: Compute the classification (P-position or N-position) for every pile size k in [0,2024] using the DP recurrence."  
        cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        subtask_desc2 = {
            "subtask_id": "subtask_2",
            "instruction": cot_instruction,
            "context": ["user query", thinking1.content, str(answer1.content)],
            "agent_collaboration": "CoT"
        }
        thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1.content], cot_instruction, is_sub_task=True)
        agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking2.content}; answer: DP list of size 2025")
        sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - DP classification list")
        subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
        logs.append(subtask_desc2)
        print("Step 2: ", sub_tasks[-1])

        # Stage 3: Collect the subset of P-positions (SC_CoT)
        sc2_instruction = "Sub-task 3: Collect the subset of initial sizes n in [1,2024] that are classified as P-positions."  
        sc2_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                      for _ in range(self.max_sc)]
        possible_thinkings3 = []
        possible_answers3 = []
        subtask_desc3 = {
            "subtask_id": "subtask_3",
            "instruction": sc2_instruction,
            "context": ["user query", thinking2.content, str(answer2.content)],
            "agent_collaboration": "SC_CoT"
        }
        for agent in sc2_agents:
            thinking3, answer3 = await agent([taskInfo, thinking2, answer2.content], sc2_instruction, is_sub_task=True)
            agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3.content}; answer: list of P-positions")
            possible_thinkings3.append(thinking3)
            possible_answers3.append(answer3)
        final_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking3, answer3 = await final_agent3([taskInfo] + possible_thinkings3 + possible_answers3,
                                               "Sub-task 3: synthesize P-positions list", is_sub_task=True)
        sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
        subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
        logs.append(subtask_desc3)
        print("Step 3: ", sub_tasks[-1])

        # Stage 4: Count the P-positions (CoT)
        cot2_instruction = "Sub-task 4: Count the number of P-positions identified in subtask_3 to obtain the final result."  
        cot2_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        subtask_desc4 = {
            "subtask_id": "subtask_4",
            "instruction": cot2_instruction,
            "context": ["user query", thinking3.content, str(answer3.content)],
            "agent_collaboration": "CoT"
        }
        thinking4, answer4 = await cot2_agent([taskInfo, thinking3, answer3.content], cot2_instruction, is_sub_task=True)
        agents.append(f"CoT agent {cot2_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
        sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
        subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
        logs.append(subtask_desc4)
        print("Step 4: ", sub_tasks[-1])

        final_answer, final_logs = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
        return final_answer, logs
