async def forward_69(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_1 = (
        "Sub-task 1: Identify and hypothesize the chemical identities of solid A, gas B, gas D, and intermediate product C by integrating all given clues including stoichiometry, physical properties (e.g., bright red color of C), reaction conditions, and hazard information. "
        "Use chemical property databases and literature to generate multiple candidate identities, score them for consistency with all clues, and select the most plausible set."
    )
    N1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, hypothesizing chemical identities, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    answer1_content = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[answer1_content]
    answer1 = answermapping_1[answer1_content]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    debate_instruction_1b = (
        "Sub-task 1b: Debate among multiple agents on the conflicting clues from Sub-task 1, especially reconciling the bright red color of product C with stoichiometric and hazard data. "
        "Agents should argue for or against candidate identities and reach a consensus or highlight uncertainties."
    )
    debate_agents_1b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1b = self.max_round
    all_thinking_1b = [[] for _ in range(N_max_1b)]
    all_answer_1b = [[] for _ in range(N_max_1b)]
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": debate_instruction_1b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1b):
        for i, agent in enumerate(debate_agents_1b):
            if r == 0:
                thinking_1b, answer_1b = await agent([taskInfo, thinking1, answer1], debate_instruction_1b, r, is_sub_task=True)
            else:
                input_infos_1b = [taskInfo, thinking1, answer1] + all_thinking_1b[r-1] + all_answer_1b[r-1]
                thinking_1b, answer_1b = await agent(input_infos_1b, debate_instruction_1b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating chemical identities, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
            all_thinking_1b[r].append(thinking_1b)
            all_answer_1b[r].append(answer_1b)
    final_decision_agent_1b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1b, answer_1b = await final_decision_agent_1b([taskInfo] + all_thinking_1b[-1] + all_answer_1b[-1], "Sub-task 1b: Make final decision on chemical identities after debate.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting chemical identities, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc1b['response'] = {
        "thinking": thinking_1b,
        "answer": answer_1b
    }
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_sc_instruction_2a = (
        "Sub-task 2a: Conduct a detailed stoichiometric and mechanistic analysis of the reaction of solid A with exactly 8 equivalents of gas B to form intermediate product C. "
        "Validate stoichiometric balance and chemical plausibility using the refined chemical identities from Sub-task 1b."
    )
    cot_sc_instruction_2b = (
        "Sub-task 2b: Analyze the reaction of intermediate product C with exactly 2 equivalents of gas D to form the extremely hazardous product E. "
        "Validate the reaction mechanism, stoichiometry, and hazard data for product E."
    )
    cot_sc_instruction_2c = (
        "Sub-task 2c: Analyze the hydrolysis of intermediate product C with water to regenerate solid A plus two different acids F (strong acid) and G (weak acid). "
        "Confirm stoichiometric and chemical consistency with previous subtasks."
    )
    cot_sc_instruction_2d = (
        "Sub-task 2d: Analyze the reaction of gas D with gas B in a 1:1 ratio to form solvent H. "
        "Validate the reaction and product identity with chemical plausibility and hazard data."
    )
    N2 = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    cot_agents_2c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    cot_agents_2d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    possible_answers_2c = []
    thinkingmapping_2c = {}
    answermapping_2c = {}
    possible_answers_2d = []
    thinkingmapping_2d = {}
    answermapping_2d = {}
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_sc_instruction_2c,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    subtask_desc2d = {
        "subtask_id": "subtask_2d",
        "instruction": cot_sc_instruction_2d,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, analyzing reaction A + 8B -> C, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, analyzing reaction C + 2D -> E, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
        thinking2c, answer2c = await cot_agents_2c[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2c[i].id}, analyzing hydrolysis of C, thinking: {thinking2c.content}; answer: {answer2c.content}")
        possible_answers_2c.append(answer2c.content)
        thinkingmapping_2c[answer2c.content] = thinking2c
        answermapping_2c[answer2c.content] = answer2c
        thinking2d, answer2d = await cot_agents_2d[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_2d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2d[i].id}, analyzing reaction D + B -> H, thinking: {thinking2d.content}; answer: {answer2d.content}")
        possible_answers_2d.append(answer2d.content)
        thinkingmapping_2d[answer2d.content] = thinking2d
        answermapping_2d[answer2d.content] = answer2d
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    answer2c_content = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking2c = thinkingmapping_2c[answer2c_content]
    answer2c = answermapping_2c[answer2c_content]
    answer2d_content = Counter(possible_answers_2d).most_common(1)[0][0]
    thinking2d = thinkingmapping_2d[answer2d_content]
    answer2d = answermapping_2d[answer2d_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {
        "thinking": thinking2c,
        "answer": answer2c
    }
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])
    sub_tasks.append(f"Sub-task 2d output: thinking - {thinking2d.content}; answer - {answer2d.content}")
    subtask_desc2d['response'] = {
        "thinking": thinking2d,
        "answer": answer2d
    }
    logs.append(subtask_desc2d)
    print("Step 2d: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = (
        "Sub-task 3: Engage in a debate and self-consistency check among multiple hypotheses generated in subtasks 1b and 2 to resolve conflicts, validate assumptions, and refine the chemical identities and reaction pathways. "
        "Reassess and update the outputs of subtasks 1b and 2 as needed to ensure coherence with all clues and chemical principles."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking_1b, answer_1b, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c, thinking2d, answer2d]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b", "thinking of subtask 2c", "answer of subtask 2c", "thinking of subtask 2d", "answer of subtask 2d"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, debating and refining chemical identities and pathways, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the chemical identities and reaction pathways for consistency and correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining chemical identities and pathways, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_reflect_instruction_4 = (
        "Sub-task 4: Analyze the chemical structure and molecular geometry of product E based on the refined chemical identity and reaction pathway from Sub-task 3. "
        "Incorporate reflexion to critically evaluate alternative structures if inconsistencies arise. Determine the symmetry elements present in E's molecular structure."
    )
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
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, analyzing molecular structure and symmetry of E, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the molecular structure and symmetry analysis of product E and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining molecular structure and symmetry analysis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = (
        "Sub-task 5: Determine the molecular symmetry group of product E by classifying its symmetry elements into one of the given point groups: C2, C2v, D4h, or D∞h. "
        "Validate that the assigned symmetry group matches one of the provided multiple-choice options."
    )
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
            agents.append(f"Debate agent {agent.id}, round {r}, determining molecular symmetry group, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the molecular symmetry group of product E.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting molecular symmetry group, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6 = (
        "Sub-task 6: Select the correct multiple-choice answer corresponding to the molecular symmetry group of E determined in Sub-task 5. "
        "If no exact match is found, apply a fallback mechanism to select the closest matching option with justification, ensuring strict compliance with the output format."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
