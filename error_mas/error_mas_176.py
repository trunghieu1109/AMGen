from collections import Counter

async def forward_176(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Correct the observed peak wavelength of Star_2 for its radial velocity (700 km/s) using the Doppler formula to find the rest-frame λ_max₂"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc0 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking0, answer0 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, correcting Doppler shift, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings.append(thinking0)
        possible_answers.append(answer0)
    final_decision_agent0 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent0(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize and choose the most consistent and correct rest-frame λ_max for Star_2. Given all the above thinking and answers, find the most consistent and correct solution for the Doppler-corrected rest-frame λ_max",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 1: Use Wien’s law on the rest-frame λ_max of both stars to determine their effective temperatures and verify that T1 = T2"
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query", thinking0.content, answer0.content], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo, thinking0, answer0], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, applying Wien's law, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings.append(thinking1)
        possible_answers.append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1(
        [taskInfo, thinking0, answer0] + possible_thinkings + possible_answers,
        "Sub-task 2: Synthesize and confirm effective temperatures are equal using Wien's law. Given all the above thinking and answers, find the most consistent temperatures and verify T1 = T2",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Apply the Stefan–Boltzmann law with R1/R2 = 1.5 and T1/T2 = 1 to compute the intrinsic luminosity ratio L1/L2"
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", thinking1.content, answer1.content], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, applying Stefan-Boltzmann law, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings.append(thinking2)
        possible_answers.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + possible_thinkings + possible_answers,
        "Sub-task 3: Synthesize and compute intrinsic luminosity ratio. Given all the above thinking and answers, find the most consistent intrinsic L1/L2",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction = "Sub-task 1: Compare the computed ratio to the provided multiple-choice options (~2.25, ~2.35, ~2.32, ~2.23) and select the closest value. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc3 = {"subtask_id": "subtask_1", "instruction": debate_instruction, "context": ["user query", thinking2.content, answer2.content], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for agent in debate_agents:
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction, r, is_sub_task=True)
            else:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2] + all_thinking[r-1] + all_answer[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking[r].append(thinking3)
            all_answer[r].append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking2, answer2] + all_thinking[-1] + all_answer[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs