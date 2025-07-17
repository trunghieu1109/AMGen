async def forward_160(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Extract and summarize the relevant information from the query, defining λ1 and λ2 and noting conditions."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    for agent in cot_agents:
        thinking_i, answer_i = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, extracting information, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize the summarized information and produce a coherent summary.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "response": {"thinking": thinking1, "answer": answer1})
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Using kinetic theory and scattering theory, derive the ratio λ2/λ1 by comparing the molecular collision cross section to the electron-molecule scattering cross section and estimate the expected numerical factor."
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    for agent in cot_agents:
        thinking_i, answer_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, deriving ratio λ2/λ1, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings2.append(thinking_i)
        possible_answers2.append(answer_i)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize the theoretical derivation and present the estimated ratio.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "response": {"thinking": thinking2, "answer": answer2})
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 3: Compare the derived ratio λ2/λ1 to the provided choices and select the matching choice."
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    for agent in cot_agents:
        thinking_i, answer_i = await agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, comparing ratio to choices, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings3.append(thinking_i)
        possible_answers3.append(answer_i)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 3: Select the correct choice that matches the theoretical prediction.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({"subtask_id": "subtask_3", "instruction": cot_sc_instruction, "response": {"thinking": thinking3, "answer": answer3})
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs