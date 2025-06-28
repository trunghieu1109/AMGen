async def forward_32(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Identify and characterize the reactants 2,5-dimethylthiophene and furan-2,5-dione by determining their detailed molecular structures, including heteroatoms, substituent positions, and stereochemical features relevant to the [4+2] cycloaddition. Provide explicit structural details such as SMILES or clear descriptions of stereochemistry."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying and characterizing reactants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Analyze the [4+2] cycloaddition reaction mechanism under thermal conditions, focusing on regioselectivity and stereoselectivity, and predict the general structural framework of the cycloadducts (endo and exo), incorporating stereochemical considerations of the reactants characterized in Sub-task 1. Include expected stereochemical outcomes and rationale."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing reaction mechanism and stereoselectivity, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3a = "Sub-task 3a: Generate explicit 3D structural representations or SMILES strings of the predicted exo and endo cycloadducts from the reaction, clearly indicating stereocenters and substituent orientations to enable unambiguous stereochemical analysis. Use the analysis from Sub-task 2."
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3a = self.max_round
    all_thinking3a = [[] for _ in range(N_max_3a)]
    all_answer3a = [[] for _ in range(N_max_3a)]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": debate_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            input_infos_3a = [taskInfo, thinking2, answer2]
            if r > 0:
                input_infos_3a.extend(all_thinking3a[r-1])
                input_infos_3a.extend(all_answer3a[r-1])
            thinking3a, answer3a = await agent(input_infos_3a, debate_instruction_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating 3D/SMILES structures of exo and endo adducts, thinking: {thinking3a.content}; answer: {answer3a.content}")
            all_thinking3a[r].append(thinking3a)
            all_answer3a[r].append(answer3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo] + all_thinking3a[-1] + all_answer3a[-1], "Sub-task 3a: Make a final decision on the generated 3D/SMILES structures of exo and endo adducts.", is_sub_task=True)
    agents.append(f"Final Decision agent on 3D/SMILES structures, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_instruction_3b = "Sub-task 3b: Systematically apply the Cahn-Ingold-Prelog (CIP) priority rules to assign R/S configurations to each stereocenter in the exo and endo adduct structures generated in Sub-task 3a. Document the stereochemical assignments in detail."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, assigning CIP R/S configurations, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_instruction_3c = "Sub-task 3c: Perform a detailed functional group and ring system analysis of the exo and endo adducts, distinguishing key features such as epoxy versus epithio bridges and their positions, to clarify structural differences relevant to product identification. Use the structures from Sub-task 3a."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_instruction_3c,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3c, answer3c = await cot_agent_3c([taskInfo, thinking3a, answer3a], cot_instruction_3c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3c.id}, analyzing functional groups and ring systems, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])

    cot_instruction_4a = "Sub-task 4a: Compare the stereochemical assignments (R/S configurations) and functional group analyses of the predicted exo product with each multiple-choice option, checking for consistency in stereochemistry, substituent positions, and ring systems. Provide detailed matching results."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3b, answer3b, thinking3c, answer3c], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, comparing stereochemistry and functional groups with options, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    reflexion_instruction_4b = "Sub-task 4b: Cross-validate the stereochemical and structural matching results from Sub-task 4a by independent reassignment or debate/reflexion to resolve any discrepancies or ambiguities, ensuring the final product choice is robust and accurate."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4b = self.max_round
    cot_inputs_4b = [taskInfo, thinking4a, answer4a]
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": reflexion_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Reflexion"
    }
    thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, reflexion_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, initial cross-validation, thinking: {thinking4b.content}; answer: {answer4b.content}")
    for i in range(N_max_4b):
        feedback, correct = await critic_agent_4b([taskInfo, thinking4b, answer4b], "Please review the stereochemical and structural matching validation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_4b.extend([thinking4b, answer4b, feedback])
        thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, reflexion_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining cross-validation, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    debate_instruction_4c = "Sub-task 4c: Review all stereochemical assignments, structural analyses, and comparison outcomes from previous subtasks to confirm the correct exo product choice from the multiple-choice options, providing a final reasoned selection."
    debate_agents_4c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4c = self.max_round
    all_thinking4c = [[] for _ in range(N_max_4c)]
    all_answer4c = [[] for _ in range(N_max_4c)]
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": debate_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4c):
        for i, agent in enumerate(debate_agents_4c):
            input_infos_4c = [taskInfo, thinking4b, answer4b]
            if r > 0:
                input_infos_4c.extend(all_thinking4c[r-1])
                input_infos_4c.extend(all_answer4c[r-1])
            thinking4c, answer4c = await agent(input_infos_4c, debate_instruction_4c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, final review and selection of exo product, thinking: {thinking4c.content}; answer: {answer4c.content}")
            all_thinking4c[r].append(thinking4c)
            all_answer4c[r].append(answer4c)
    final_decision_agent_4c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4c, answer4c = await final_decision_agent_4c([taskInfo] + all_thinking4c[-1] + all_answer4c[-1], "Sub-task 4c: Make a final decision on the correct exo product choice.", is_sub_task=True)
    agents.append(f"Final Decision agent on correct exo product, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {
        "thinking": thinking4c,
        "answer": answer4c
    }
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4c, answer4c, sub_tasks, agents)
    return final_answer, logs
