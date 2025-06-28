async def forward_173(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify and define all given physical quantities and parameters from the problem statement, including the initial nucleus rest mass M, initial rest-mass energy (300 GeV), the mass ratio of the two fragments (more massive fragment is twice the rest mass of the lighter one), and the total rest mass of the fragments after fission (99% of M)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying physical quantities, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Express the rest masses of the two fragments symbolically in terms of M and the given mass ratio, then calculate their explicit rest masses as m1 (more massive fragment) and m2 (lighter fragment), ensuring m1 + m2 = 0.99 M and m1 = 2 m2."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating fragment rest masses, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Sub-task 3: Verify the total rest mass of the two fragments equals 99% of the initial mass M, confirming the mass defect and the energy released in the fission process, based on outputs from Sub-task 2."
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
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, verifying total rest mass, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "Please review the total rest mass verification and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining total rest mass verification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4_1 = "Sub-task 4_1: Write down the relativistic conservation equations explicitly: conservation of momentum (p1 = -p2) and conservation of total energy (initial rest energy = sum of fragment total energies), using relativistic momentum p = gamma * m * v and total energy E = gamma * m * c^2."
    cot_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4_1 = []
    thinkingmapping_4_1 = {}
    answermapping_4_1 = {}
    subtask_desc4_1 = {
        "subtask_id": "subtask_4_1",
        "instruction": cot_sc_instruction_4_1,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking4_1, answer4_1 = await cot_agents_4_1[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4_1[i].id}, writing relativistic conservation equations, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
        possible_answers_4_1.append(answer4_1.content)
        thinkingmapping_4_1[answer4_1.content] = thinking4_1
        answermapping_4_1[answer4_1.content] = answer4_1
    answer4_1_content = Counter(possible_answers_4_1).most_common(1)[0][0]
    thinking4_1 = thinkingmapping_4_1[answer4_1_content]
    answer4_1 = answermapping_4_1[answer4_1_content]
    sub_tasks.append(f"Sub-task 4_1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {"thinking": thinking4_1, "answer": answer4_1}
    logs.append(subtask_desc4_1)
    print("Step 4_1: ", sub_tasks[-1])
    
    cot_sc_instruction_4_2 = "Sub-task 4_2: Solve the relativistic conservation equations symbolically or numerically to find the common magnitude of momentum p of the fragments and their velocities v1 and v2, ensuring consistency with relativistic definitions."
    cot_agents_4_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4_2 = []
    thinkingmapping_4_2 = {}
    answermapping_4_2 = {}
    subtask_desc4_2 = {
        "subtask_id": "subtask_4_2",
        "instruction": cot_sc_instruction_4_2,
        "context": ["user query", "thinking of subtask 4_1", "answer of subtask 4_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking4_2, answer4_2 = await cot_agents_4_2[i]([taskInfo, thinking4_1, answer4_1], cot_sc_instruction_4_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4_2[i].id}, solving relativistic equations, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
        possible_answers_4_2.append(answer4_2.content)
        thinkingmapping_4_2[answer4_2.content] = thinking4_2
        answermapping_4_2[answer4_2.content] = answer4_2
    answer4_2_content = Counter(possible_answers_4_2).most_common(1)[0][0]
    thinking4_2 = thinkingmapping_4_2[answer4_2_content]
    answer4_2 = answermapping_4_2[answer4_2_content]
    sub_tasks.append(f"Sub-task 4_2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {"thinking": thinking4_2, "answer": answer4_2}
    logs.append(subtask_desc4_2)
    print("Step 4_2: ", sub_tasks[-1])
    
    cot_reflect_instruction_4_3 = "Sub-task 4_3: Calculate the relativistic kinetic energy T1_rel of the more massive fragment using T1_rel = E1 - m1 c^2, where E1 = sqrt(p^2 c^2 + m1^2 c^4). Provide a numeric value for T1_rel in MeV or GeV, based on outputs from Sub-task 4_2."
    cot_agent_4_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4_3 = self.max_round
    cot_inputs_4_3 = [taskInfo, thinking4_2, answer4_2]
    subtask_desc4_3 = {
        "subtask_id": "subtask_4_3",
        "instruction": cot_reflect_instruction_4_3,
        "context": ["user query", "thinking of subtask 4_2", "answer of subtask 4_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking4_3, answer4_3 = await cot_agent_4_3(cot_inputs_4_3, cot_reflect_instruction_4_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_3.id}, calculating relativistic kinetic energy T1_rel, thinking: {thinking4_3.content}; answer: {answer4_3.content}")
    for i in range(N_max_4_3):
        feedback, correct = await critic_agent_4_3([taskInfo, thinking4_3, answer4_3], "Verify that T1_rel and momentum p are explicitly computed and numeric, and calculations are correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4_3.extend([thinking4_3, answer4_3, feedback])
        thinking4_3, answer4_3 = await cot_agent_4_3(cot_inputs_4_3, cot_reflect_instruction_4_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_3.id}, refining relativistic kinetic energy calculation, thinking: {thinking4_3.content}; answer: {answer4_3.content}")
    sub_tasks.append(f"Sub-task 4_3 output: thinking - {thinking4_3.content}; answer - {answer4_3.content}")
    subtask_desc4_3['response'] = {"thinking": thinking4_3, "answer": answer4_3}
    logs.append(subtask_desc4_3)
    print("Step 4_3: ", sub_tasks[-1])
    
    cot_instruction_5 = "Sub-task 5: Calculate the classical (non-relativistic) kinetic energy T1_classical of the more massive fragment using the momentum p found in Sub-task 4_2 and the classical formula T1_classical = p^2 / (2 m1), ensuring precise numeric evaluation without heuristic approximations."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4_2", "answer of subtask 4_2"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4_2, answer4_2], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating classical kinetic energy T1_classical, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_sc_instruction_6 = "Sub-task 6: Compute the difference \u0394T = T1_rel - T1_classical strictly from the numeric values obtained in Sub-tasks 4_3 and 5, avoiding any estimation or guesswork."
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", "thinking of subtask 4_3", "answer of subtask 4_3", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking4_3, answer4_3, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, computing difference between T1_rel and T1_classical, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    answer6_content = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[answer6_content]
    answer6 = answermapping_6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    debate_instruction_7 = "Sub-task 7: Compare the computed difference \u0394T with the provided multiple-choice options (10 MeV, 5 MeV, 2 MeV, 20 MeV) and select the correct answer choice (A, B, C, or D) based on the closest match."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct answer choice, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct answer choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_reflect_instruction_8 = "Sub-task 8: Perform a verification and consistency check of the relativistic kinetic energy calculation (subtask_4_3) and the difference calculation (subtask_6) to ensure no computational or conceptual errors before finalizing the answer."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking4_3, answer4_3, thinking6, answer6]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", "thinking of subtask 4_3", "answer of subtask 4_3", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, verifying calculations consistency, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback, correct = await critic_agent_8([taskInfo, thinking8, answer8], "Verify the correctness and consistency of relativistic kinetic energy and difference calculations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining verification, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs