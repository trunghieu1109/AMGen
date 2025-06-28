async def forward_136(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Identify and list all given physical parameters and constants from the problem statement, including initial kinetic energy (T ≈ 0), final kinetic energy (T1 = 950 MeV), voltage amplitude (U0 = 190 kV), initial phase (Φ0 = π/4), proton charge (q), proton rest mass (m0), speed of light (c), and magnetic field strength (B) if available or to be assumed. Ensure units are consistent and clearly stated."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting physical parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    debate_instruction_2 = "Sub-task 2: Explain the working principle of a synchrocyclotron, focusing on how the proton gains kinetic energy through successive accelerations at the gap crossing, the role of the alternating voltage U(t) = U0 cos[2πν(α)t + Φ0], and how the frequency ν(α) depends on relativistic effects and the proton's energy (including the formula ν(α) = qB / (2πγm0)). Clarify the meaning of two accelerations in the problem context."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            input_infos_2 = [taskInfo, thinking1, answer1]
            if r > 0:
                input_infos_2.extend(all_thinking2[r-1])
            thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, explaining synchrocyclotron principle, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Make a final decision on the working principle of the synchrocyclotron.", is_sub_task=True)
    agents.append(f"Final Decision agent on synchrocyclotron principle, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])
    cot_sc_instruction_3a = "Sub-task 3a: Derive the proton's phase at each gap crossing explicitly, incorporating the initial phase Φ0 and the time-dependent frequency ν(α), to determine the phase at the nth revolution."
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, deriving proton phase at nth gap crossing, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    most_common_answer_3a = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[most_common_answer_3a]
    answer3a = answermapping_3a[most_common_answer_3a]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Subtask 3a answer: ", sub_tasks[-1])
    cot_sc_instruction_3b = "Sub-task 3b: Formulate the energy gain per revolution ΔE_n as ΔE_n = q U0 cos(Φ0 + Δφ_n), where Δφ_n is the phase increment at the nth crossing, avoiding unjustified averaging over a full cycle."
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking3a, answer3a], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, formulating energy gain per revolution, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    most_common_answer_3b = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[most_common_answer_3b]
    answer3b = answermapping_3b[most_common_answer_3b]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Subtask 3b answer: ", sub_tasks[-1])
    cot_sc_instruction_3c = "Sub-task 3c: Sum the incremental energy gains ΔE_n over successive revolutions until the total kinetic energy gain reaches T1 = 950 MeV, accounting for relativistic mass increase and frequency tuning."
    cot_agents_3c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3c = []
    thinkingmapping_3c = {}
    answermapping_3c = {}
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_sc_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3c, answer3c = await cot_agents_3c[i]([taskInfo, thinking3b, answer3b], cot_sc_instruction_3c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3c[i].id}, summing incremental energy gains, thinking: {thinking3c.content}; answer: {answer3c.content}")
        possible_answers_3c.append(answer3c.content)
        thinkingmapping_3c[answer3c.content] = thinking3c
        answermapping_3c[answer3c.content] = answer3c
    most_common_answer_3c = Counter(possible_answers_3c).most_common(1)[0][0]
    thinking3c = thinkingmapping_3c[most_common_answer_3c]
    answer3c = answermapping_3c[most_common_answer_3c]
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Subtask 3c answer: ", sub_tasks[-1])
    cot_reflect_instruction_3d = "Sub-task 3d: Validate the derived energy gain per revolution and total number of revolutions by checking units, physical plausibility, and consistency with synchrocyclotron operation principles."
    cot_agent_3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3d = self.max_round
    cot_inputs_3d = [taskInfo, thinking3c, answer3c]
    subtask_desc3d = {
        "subtask_id": "subtask_3d",
        "instruction": cot_reflect_instruction_3d,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Reflexion"
    }
    thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, validating energy gain and revolutions, thinking: {thinking3d.content}; answer: {answer3d.content}")
    for i in range(N_max_3d):
        feedback, correct = await critic_agent_3d([taskInfo, thinking3d, answer3d], "Please review the validation of energy gain per revolution and total revolutions, and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_3d.extend([thinking3d, answer3d, feedback])
        thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, refining validation, thinking: {thinking3d.content}; answer: {answer3d.content}")
    sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d.content}; answer - {answer3d.content}")
    subtask_desc3d['response'] = {"thinking": thinking3d, "answer": answer3d}
    logs.append(subtask_desc3d)
    print("Subtask 3d answer: ", sub_tasks[-1])
    cot_sc_instruction_4 = "Sub-task 4: Calculate the total number of revolutions N required for the proton to reach the kinetic energy T1 = 950 MeV by dividing the total energy gain by the validated energy gain per revolution, incorporating relativistic corrections and initial phase effects."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3d", "answer of subtask 3d"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3d, answer3d], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, calculating total revolutions, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[most_common_answer_4]
    answer4 = answermapping_4[most_common_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Subtask 4 answer: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Compare the calculated number of revolutions N with the provided multiple-choice options (1864, 2500, 3536, 5300). Critically evaluate discrepancies and select the closest physically consistent answer choice, providing justification for the selection."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating closest answer choice, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the closest physically consistent number of revolutions and select the corresponding letter choice.", is_sub_task=True)
    agents.append(f"Final Decision agent on final choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Subtask 5 answer: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs