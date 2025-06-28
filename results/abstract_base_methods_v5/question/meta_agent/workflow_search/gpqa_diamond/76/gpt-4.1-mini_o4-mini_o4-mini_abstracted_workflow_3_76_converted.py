async def forward_76(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = (
        "Sub-task 1a: Elucidate detailed mechanistic pathways for the first reaction (((3-methylbut-2-en-1-yl)oxy)methyl)benzene + (1. BuLi, 2. H+), "
        "explicitly number atoms, consider both [1,2]-Wittig and [2,3]-sigmatropic rearrangement mechanisms, include intermediate structures and electron flow."
    )
    debate_agents_1a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1a = self.max_round
    all_thinking_1a = [[] for _ in range(N_max_1a)]
    all_answer_1a = [[] for _ in range(N_max_1a)]
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1a):
        for i, agent in enumerate(debate_agents_1a):
            input_infos_1a = [taskInfo]
            if r > 0:
                input_infos_1a.extend(all_thinking_1a[r-1])
            thinking_1a, answer_1a = await agent(input_infos_1a, cot_instruction_1a, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, elucidating mechanisms for first reaction, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
            all_thinking_1a[r].append(thinking_1a)
            all_answer_1a[r].append(answer_1a)
    final_decision_agent_1a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1a, answer_1a = await final_decision_agent_1a([taskInfo] + all_thinking_1a[-1] + all_answer_1a[-1], "Sub-task 1a: Make final decision on mechanistic pathways for first reaction.", is_sub_task=True)
    agents.append(f"Final Decision agent on mechanistic pathways, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = (
        "Sub-task 1b: Determine possible product structures and stereochemistry for product A from each candidate mechanism identified in subtask_1a, "
        "specify alkene position, double bond geometry (E/Z), functional groups with explicit atom numbering."
    )
    N_1b = self.max_sc
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_1b)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1b):
        thinking_1b, answer_1b = await cot_agents_1b[i]([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, determining product structures for A, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
        possible_answers_1b.append(answer_1b.content)
        thinkingmapping_1b[answer_1b.content] = thinking_1b
        answermapping_1b[answer_1b.content] = answer_1b
    most_common_answer_1b = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking_1b = thinkingmapping_1b[most_common_answer_1b]
    answer_1b = answermapping_1b[most_common_answer_1b]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = (
        "Sub-task 1c: Evaluate plausibility of each candidate product for A by comparing mechanistic consistency, reaction conditions, and literature precedents, "
        "select the most likely major product with detailed justification."
    )
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "CoT"
    }
    thinking_1c, answer_1c = await cot_agent_1c([taskInfo, thinking_1b, answer_1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, evaluating plausibility of product A candidates, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    debate_instruction_2a = (
        "Sub-task 2a: Analyze the Cope rearrangement of 3,4,5,7,8,9-hexamethyl-1,11-dimethylene-2,6,10,11,11a,11b-hexahydro-1H-benzo[cd]indeno[7,1-gh]azulene under heat, "
        "explicitly number ring carbons, map migrating π-bonds and sigma bonds, detail which carbons become saturated or remain unsaturated."
    )
    debate_agents_2a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2a = self.max_round
    all_thinking_2a = [[] for _ in range(N_max_2a)]
    all_answer_2a = [[] for _ in range(N_max_2a)]
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": debate_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2a):
        for i, agent in enumerate(debate_agents_2a):
            input_infos_2a = [taskInfo]
            if r > 0:
                input_infos_2a.extend(all_thinking_2a[r-1])
            thinking_2a, answer_2a = await agent(input_infos_2a, debate_instruction_2a, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing Cope rearrangement mechanism, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
            all_thinking_2a[r].append(thinking_2a)
            all_answer_2a[r].append(answer_2a)
    final_decision_agent_2a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2a, answer_2a = await final_decision_agent_2a([taskInfo] + all_thinking_2a[-1] + all_answer_2a[-1], "Sub-task 2a: Make final decision on Cope rearrangement mechanistic analysis.", is_sub_task=True)
    agents.append(f"Final Decision agent on Cope rearrangement mechanism, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_instruction_2b = (
        "Sub-task 2b: Propose detailed candidate structures for product B from the Cope rearrangement, specify degree of saturation (tetrahydro vs hexahydro), "
        "double bond positions, and stereochemistry, supported by mechanistic rationale."
    )
    N_2b = self.max_sc
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_2b)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_2b):
        thinking_2b, answer_2b = await cot_agents_2b[i]([taskInfo, thinking_2a, answer_2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, proposing candidate structures for product B, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
        possible_answers_2b.append(answer_2b.content)
        thinkingmapping_2b[answer_2b.content] = thinking_2b
        answermapping_2b[answer_2b.content] = answer_2b
    most_common_answer_2b = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[most_common_answer_2b]
    answer_2b = answermapping_2b[most_common_answer_2b]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_instruction_2c = (
        "Sub-task 2c: Critically evaluate and compare candidate product structures for B against mechanistic feasibility and reaction conditions, "
        "select the most plausible major product with detailed justification."
    )
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_instruction_2c,
        "context": ["user query", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "CoT"
    }
    thinking_2c, answer_2c = await cot_agent_2c([taskInfo, thinking_2b, answer_2b], cot_instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, evaluating plausibility of product B candidates, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 2c: ", sub_tasks[-1])
    
    cot_instruction_3 = (
        "Sub-task 3: Systematically compare the selected candidate products A and B from subtasks 1c and 2c with the four given multiple-choice options, "
        "matching exact structural features, atom numbering, double bond positions, and stereochemistry to identify exact matches or closest correspondences."
    )
    N_3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c", "thinking of subtask_2c", "answer of subtask_2c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_3):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_1c, answer_1c, thinking_2c, answer_2c], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, comparing candidate products with choices, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3.content)
        thinkingmapping_3[answer_3.content] = thinking_3
        answermapping_3[answer_3.content] = answer_3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking_3 = thinkingmapping_3[most_common_answer_3]
    answer_3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instruction_4 = (
        "Sub-task 4: Select the final major product choice (A, B, C, or D) that best corresponds to the mechanistically justified products A and B, "
        "ensuring consistency and correctness of stereochemical and structural assignments."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking_3, answer_3]
            if r > 0:
                input_infos_4.extend(all_thinking_4[r-1])
            thinking_4, answer_4 = await agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting final major product, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Make final decision on major product selection.", is_sub_task=True)
    agents.append(f"Final Decision agent on major product selection, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_reflect_instruction_5 = (
        "Sub-task 5: Perform a reflexive review of the entire reasoning chain and final selection by cross-checking mechanistic rules, "
        "stereochemical assignments, and structural matches to detect and correct any inconsistencies or errors before finalizing the answer."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b, thinking_1c, answer_1c, thinking_2a, answer_2a, thinking_2b, answer_2b, thinking_2c, answer_2c, thinking_3, answer_3, thinking_4, answer_4]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking and answers of subtasks 1a-4"],
        "agent_collaboration": "Reflexion"
    }
    thinking_5, answer_5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, reviewing entire reasoning chain, thinking: {thinking_5.content}; answer: {answer_5.content}")
    for i in range(N_max_5):
        feedback_5, correct_5 = await critic_agent_5([taskInfo, thinking_5, answer_5], "Please review the reasoning and final selection for inconsistencies or errors.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback_5.content}; answer: {correct_5.content}")
        if correct_5.content.strip().lower() == "true":
            break
        cot_inputs_5.extend([thinking_5, answer_5, feedback_5])
        thinking_5, answer_5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining reasoning chain, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
