async def forward_172(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Identify and list all given physical quantities and constants from the problem statement, including electron speed (v = 2 × 10^8 m/s), uncertainty in position (Δx = 0.1 nm), and relevant physical constants such as Planck's constant (ħ) and electron mass (m)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identify physical quantities, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Recall and write down the relevant physical principles and formulas needed to relate uncertainty in position (Δx) to uncertainty in momentum (Δp) using Heisenberg's uncertainty principle, and then relate Δp to uncertainty in energy (ΔE) using two expressions: (a) ΔE_linear ≈ v × Δp, and (b) ΔE_bound ≥ (Δp)^2 / (2m)."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, recall formulas, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_reflect_instruction_3 = "Sub-task 3: Calculate the minimum uncertainty in momentum (Δp) of the electron using the uncertainty in position (Δx) and Heisenberg's uncertainty principle (Δx × Δp ≥ ħ/2)."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, calculate Δp, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the calculation of minimum uncertainty in momentum Δp and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining Δp calculation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction_4a = "Sub-task 4a: Calculate the energy uncertainty estimate ΔE_linear = v × Δp using the electron speed and the momentum uncertainty from Subtask 3."
    debate_instruction_4b = "Sub-task 4b: Calculate the energy uncertainty lower bound ΔE_bound = (Δp)^2 / (2m) using the momentum uncertainty from Subtask 3 and the electron mass."
    debate_agents_4a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    debate_agents_4b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4a = [[] for _ in range(N_max_4)]
    all_answer4a = [[] for _ in range(N_max_4)]
    all_thinking4b = [[] for _ in range(N_max_4)]
    all_answer4b = [[] for _ in range(N_max_4)]
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": debate_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": debate_instruction_4b,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4a):
            if r == 0:
                thinking4a, answer4a = await agent([taskInfo, thinking3, answer3], debate_instruction_4a, r, is_sub_task=True)
            else:
                input_infos_4a = [taskInfo, thinking3, answer3] + all_thinking4a[r-1] + all_answer4a[r-1]
                thinking4a, answer4a = await agent(input_infos_4a, debate_instruction_4a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculate ΔE_linear, thinking: {thinking4a.content}; answer: {answer4a.content}")
            all_thinking4a[r].append(thinking4a)
            all_answer4a[r].append(answer4a)
        for i, agent in enumerate(debate_agents_4b):
            if r == 0:
                thinking4b, answer4b = await agent([taskInfo, thinking3, answer3], debate_instruction_4b, r, is_sub_task=True)
            else:
                input_infos_4b = [taskInfo, thinking3, answer3] + all_thinking4b[r-1] + all_answer4b[r-1]
                thinking4b, answer4b = await agent(input_infos_4b, debate_instruction_4b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculate ΔE_bound, thinking: {thinking4b.content}; answer: {answer4b.content}")
            all_thinking4b[r].append(thinking4b)
            all_answer4b[r].append(answer4b)
    final_decision_agent_4c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4c, answer4c = await final_decision_agent_4c(
        [taskInfo] + all_thinking4a[-1] + all_answer4a[-1] + all_thinking4b[-1] + all_answer4b[-1],
        "Sub-task 4c: Compare the two energy uncertainty values ΔE_linear and ΔE_bound, and select the larger value as the minimum uncertainty in energy ΔE_min, ensuring the correct theoretical lower bound is chosen.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, selecting ΔE_min, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4a and 4b output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": "Sub-task 4c: Compare ΔE_linear and ΔE_bound and select the larger as ΔE_min.",
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Final Decision"
    }
    subtask_desc4c['response'] = {
        "thinking": thinking4c,
        "answer": answer4c
    }
    logs.append(subtask_desc4c)
    print("Step 4: ", sub_tasks[-1])
    reflexion_instruction_5 = "Sub-task 5: Cross-check the calculated minimum uncertainty in energy ΔE_min against the provided multiple-choice options (~10^(-19) J, ~10^(-18) J, ~10^(-16) J, ~10^(-17) J), identify the closest matching option, and select the corresponding letter choice (A, B, C, or D)."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4c, answer4c]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_5,
        "context": ["user query", "thinking of subtask 4c", "answer of subtask 4c"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, cross-check ΔE_min with choices, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please review the matching of ΔE_min to multiple-choice options and correctness of final choice.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining final choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs