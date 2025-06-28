async def forward_95(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Convert the given angular size θ = 10^-17 degrees into radians with explicit numeric evaluation and detailed exponent arithmetic to enable accurate physical size calculations."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, convert angular size from degrees to radians with numeric evaluation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    debate_instruction_2 = "Sub-task 2: Calculate the physical diameter D of the black hole's event horizon using the angular size in radians from Subtask 1 and the given distance d = 10^10 parsecs, performing full numeric evaluation with units and exponent tracking. Debate the correctness and assumptions involved."
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
            agents.append(f"Debate agent {agent.id}, round {r}, calculating physical diameter, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Make a final decision on physical diameter calculation.", is_sub_task=True)
    agents.append(f"Final Decision agent on physical diameter, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction_3 = "Sub-task 3: Estimate the Schwarzschild radius R_s of the black hole by interpreting the physical diameter D from Subtask 2 as twice the Schwarzschild radius, ensuring unit consistency and numeric precision. Use self-consistency chain-of-thought to verify calculations."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, estimating Schwarzschild radius, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_reflect_instruction_4 = "Sub-task 4: Calculate the mass M of the black hole from the Schwarzschild radius R_s obtained in Subtask 3 using the formula R_s = 2GM/c^2, with explicit numeric evaluation and exponent tracking. Refine the calculation through reflexion."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, initial mass calculation, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the mass calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refined mass calculation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Perform a physical plausibility check on the Schwarzschild radius R_s and mass M calculated in Subtasks 3 and 4, comparing them with typical supermassive black hole scales to validate assumptions and detect possible errors. Debate the plausibility and assumptions."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4]
            if r > 0:
                input_infos_5.extend(all_thinking5[r-1])
            thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, physical plausibility check, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make a final decision on physical plausibility.", is_sub_task=True)
    agents.append(f"Final Decision agent on physical plausibility, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction_6 = "Sub-task 6: Compute the event horizon area A = 4πR_s^2 using the Schwarzschild radius R_s from Subtask 3, with detailed numeric evaluation and careful exponent arithmetic."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, computing event horizon area, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_sc_instruction_7 = "Sub-task 7: Calculate the entropy S of the black hole using the Bekenstein-Hawking formula S = k_B * A / (4 * l_p^2), where k_B is Boltzmann's constant and l_p is the Planck length, performing explicit numeric evaluation with step-by-step exponent tracking to ensure accurate order-of-magnitude estimation. Use self-consistency chain-of-thought to verify results."
    N = self.max_sc
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_7 = []
    thinkingmapping_7 = {}
    answermapping_7 = {}
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking6, answer6], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, calculating entropy, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7.content)
        thinkingmapping_7[answer7.content] = thinking7
        answermapping_7[answer7.content] = answer7
    most_common_answer_7 = Counter(possible_answers_7).most_common(1)[0][0]
    thinking7 = thinkingmapping_7[most_common_answer_7]
    answer7 = answermapping_7[most_common_answer_7]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_reflect_instruction_8 = "Sub-task 8: Conduct a sanity check and reflexion on the computed entropy S from Subtask 7, verifying the physical plausibility of the result and consistency with known astrophysical black hole entropy magnitudes; if inconsistencies arise, revisit prior calculations."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking7, answer7]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, initial entropy sanity check, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback, correct = await critic_agent_8([taskInfo, thinking8, answer8], "please review the entropy sanity check and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refined entropy sanity check, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    cot_sc_instruction_9 = "Sub-task 9: Determine the order of magnitude of the entropy S from Subtask 7 and select the closest matching option from the given multiple-choice answers (A: 10^59 J/K, B: 10^65 J/K, C: 10^62 J/K, D: 10^66 J/K), providing the final answer as a single letter choice using self-consistency chain-of-thought."
    N = self.max_sc
    cot_agents_9 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_9 = []
    thinkingmapping_9 = {}
    answermapping_9 = {}
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_sc_instruction_9,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7", "thinking of subtask 8", "answer of subtask 8"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking9, answer9 = await cot_agents_9[i]([taskInfo, thinking7, answer7, thinking8, answer8], cot_sc_instruction_9, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_9[i].id}, determining order of magnitude and selecting closest option, thinking: {thinking9.content}; answer: {answer9.content}")
        possible_answers_9.append(answer9.content)
        thinkingmapping_9[answer9.content] = thinking9
        answermapping_9[answer9.content] = answer9
    most_common_answer_9 = Counter(possible_answers_9).most_common(1)[0][0]
    thinking9 = thinkingmapping_9[most_common_answer_9]
    answer9 = answermapping_9[most_common_answer_9]
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs