async def forward_165(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1a = "Sub-task 1a: Identify all relevant fields and particles from the given Lagrangian, including singlet fermions (N_iR), scalar doublet (S), and singlet scalar (φ). Specify their gauge representations and quantum numbers explicitly."
    cot_instruction_1b = "Sub-task 1b: Describe the interactions among the identified fields and explicitly list the vacuum expectation values (VEVs) x = ⟨φ⟩ and v = ⟨h⟩, emphasizing their roles in spontaneous symmetry breaking."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, identifying fields and quantum numbers, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {
        "thinking": thinking1a,
        "answer": answer1a
    }
    logs.append(subtask_desc1a)
    cot_instruction_1b_full = cot_instruction_1b + " Use the output from Sub-task 1a to detail interactions and VEVs explicitly."
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b_full,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking1a, answer1a], cot_instruction_1b_full, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, detailing interactions and VEVs, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {
        "thinking": thinking1b,
        "answer": answer1b
    }
    logs.append(subtask_desc1b)
    cot_sc_instruction_2 = "Sub-task 2: Analyze the origin and physical role of the pseudo-Goldstone boson H_2 in the model, detailing how it arises from spontaneous symmetry breaking of the scalar fields φ and h, and explain the mechanism by which radiative corrections generate a nonzero mass for H_2. Use self-consistency by generating multiple perspectives."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing pseudo-Goldstone boson H_2, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_instruction_3 = "Sub-task 3: Derive or recall the general formula for the one-loop radiative correction to the mass squared of the pseudo-Goldstone boson H_2. Explicitly enumerate contributions from all relevant particles: scalar bosons (H_1, H^±, H^0, A^0), gauge bosons (W, Z), the top quark (t), and singlet fermions (N_i). Clearly specify the sign and dependence of each contribution and the role of the combined VEV factor (x^2 + v^2) in the denominator. Use a Debate format to ensure completeness and correctness, including a validation step to check for missing terms."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], cot_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, cot_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deriving formula for radiative correction, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Validate completeness and correctness of the derived formula for the radiative correction mass squared of H_2.", is_sub_task=True)
    agents.append(f"Final Decision agent, validating formula completeness, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    debate_instruction_4 = "Sub-task 4: Compare each given multiple-choice formula for M_h_2^2 against the validated general formula from Sub-task 3. Verify presence, sign, and completeness of all particle contributions, correct placement of the VEV factor (x^2 + v^2), and consistency of coefficients. Explicitly note any missing or incorrect terms. Use a hybrid approach combining Self-Consistency CoT and Reflexion to catch omissions and ensure completeness before final selection."
    N = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT + Reflexion"
    }
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agents_4[0](cot_inputs_4, debate_instruction_4, 0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agents_4[0].id}, initial comparison of formulas, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the formula comparison and note any missing or incorrect terms.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agents_4[0](cot_inputs_4, debate_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"CoT agent {cot_agents_4[0].id}, refining formula comparison, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    debate_instruction_5 = "Sub-task 5: Select the correct approximation formula for the mass squared of the pseudo-Goldstone boson H_2 through radiative corrections from the given choices, based on the detailed validation in Sub-task 4. Confirm symbolic completeness and physical consistency rather than numerical evaluation, given the absence of explicit parameter values. Use a Debate format to finalize the selection."
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
            agents.append(f"Debate agent {agent.id}, round {r}, selecting final formula, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the approximate mass squared formula of the pseudo-Goldstone boson H_2.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing formula selection, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
