async def forward_69(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = (
        "Sub-task 1a: Identify gases B and D by analyzing their 1:1 reaction to form solvent H, "
        "using stoichiometric ratios, chemical properties, and solvent characteristics to propose chemically plausible identities consistent with the reaction context. "
        "Explicitly state assumptions and verify stoichiometric and chemical consistency."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, identifying gases B and D, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = (
        "Sub-task 1b: Identify solid A and product C by applying the stoichiometric ratio of 1:8 (A:B), the bright red color of C, "
        "and the acid strengths of F and G formed upon hydrolysis of C. Use identities of B and D from Sub-task 1a, verify chemical plausibility and consistency. "
        "Explicitly state assumptions and verify stoichiometric and chemical consistency."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, identifying solid A and product C, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    N_sc_2 = self.max_sc
    cot_sc_instruction_2 = (
        "Sub-task 2: Identify the hazardous product E formed from the reaction of C with 2 equivalents of D, "
        "using the confirmed identities of C and D from Sub-task 1b and 1a, stoichiometry, and known hazardous compounds. "
        "Generate multiple hypotheses and evaluate them for chemical plausibility and consistency with prior subtasks. "
        "Explicitly state assumptions and verify stoichiometric and chemical consistency."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying hazardous product E, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2.content)
        thinkingmapping_2[answer_2.content] = thinking_2
        answermapping_2[answer_2.content] = answer_2
    answer_2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking_2 = thinkingmapping_2[answer_2_content]
    answer_2 = answermapping_2[answer_2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = (
        "Sub-task 3: Analyze the molecular structure of product E to determine its molecular geometry and identify all symmetry elements, "
        "using formal point-group determination methods and verified structural data from Sub-task 2. "
        "Iteratively refine the analysis based on critic feedback to ensure accuracy and completeness."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b, thinking_2, answer_2]
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "thinking of subtask 1b", "answer of subtask 1b", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, analyzing molecular geometry and symmetry of product E, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking_3, answer_3], "Please review the molecular geometry and symmetry analysis and provide any limitations or corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback])
        thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining molecular symmetry analysis, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instruction_4 = (
        "Sub-task 4: Compare the identified molecular symmetry group of product E with the given multiple-choice options (C2, C2v, D4h, D∞h). "
        "Evaluate the best match using a systematic scoring of symmetry elements and select the correct answer in the required output format (single letter without punctuation). "
        "Use debate among agents with quantitative tie-breakers to finalize the choice."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_3, answer_3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_3, answer_3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing molecular symmetry groups and selecting correct answer, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Make final decision on the molecular symmetry group of product E.", is_sub_task=True)
    agents.append(f"Final Decision agent on molecular symmetry group selection, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs