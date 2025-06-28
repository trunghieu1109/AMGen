async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agent_base = LLMAgentBase
    N_sc = self.max_sc
    N_round = self.max_round
    debate_roles = self.debate_role
    
    cot_instruction_1 = "Sub-task 1: Perform detailed structural analysis of all reactants in reactions A and B. Provide explicit atom numbering, identify α,β-unsaturation and carbonyl groups relevant to Michael addition, and generate skeletal structures or SMILES strings to anchor subsequent mapping."
    cot_agent_1 = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, detailed structural analysis of reactants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2a = "Sub-task 2: Using the structural data from Sub-task 1, map nucleophilic and electrophilic sites on the reactants of reaction A. Explicitly identify the β-carbon for nucleophilic attack, confirm regiochemistry and stereochemistry possibilities, and provide atom mapping details."
    cot_agents_2a = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc2a = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    for i in range(N_sc):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1], cot_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, mapping reactive sites reaction A, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_2b = "Sub-task 3: Using the structural data from Sub-task 1, map nucleophilic and electrophilic sites on the reactants of reaction B. Explicitly identify the β-carbon for nucleophilic attack, confirm regiochemistry and stereochemistry possibilities, and provide atom mapping details."
    cot_agents_2b = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc2b = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    for i in range(N_sc):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking1, answer1], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, mapping reactive sites reaction B, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_3a = "Sub-task 4: Predict the product structure of reaction A by applying the Michael addition mechanism with explicit atom mapping from reactants to product. Illustrate new bond formation, substitution position, and stereochemistry. Validate the predicted structure by cross-checking connectivity and atom numbering. Provide text-based sketches or atom maps before IUPAC naming."
    cot_agent_3a = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = cot_agent_base(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_3a = [taskInfo, thinking2a, answer2a]
    thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, predicting product structure of reaction A, thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_round):
        feedback3a, correct3a = await critic_agent_3a([taskInfo, thinking3a, answer3a], "Please review the predicted product structure of reaction A, focusing on atom mapping, stereochemistry, substitution positions, and connectivity. Provide limitations if any.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, feedback on reaction A product, thinking: {feedback3a.content}; answer: {correct3a.content}")
        if correct3a.content == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback3a])
        thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining product structure of reaction A, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_3b = "Sub-task 5: Predict the product structure of reaction B by applying the Michael addition mechanism with explicit atom mapping from reactants to product. Illustrate new bond formation, substitution position, and stereochemistry. Validate the predicted structure by cross-checking connectivity and atom numbering. Provide text-based sketches or atom maps before IUPAC naming."
    cot_agent_3b = cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = cot_agent_base(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_3b = [taskInfo, thinking2b, answer2b]
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, predicting product structure of reaction B, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_round):
        feedback3b, correct3b = await critic_agent_3b([taskInfo, thinking3b, answer3b], "Please review the predicted product structure of reaction B, focusing on atom mapping, stereochemistry, substitution positions, and connectivity. Provide limitations if any.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, feedback on reaction B product, thinking: {feedback3b.content}; answer: {correct3b.content}")
        if correct3b.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback3b])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining product structure of reaction B, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 6: Generate accurate IUPAC names for the predicted products of reactions A and B based on the validated structures from Subtasks 4 and 5. Ensure correct substitution positions, stereochemistry, and functional groups are reflected. Use multiple agents independently to generate names, then apply voting or debate to resolve discrepancies."
    cot_agents_4 = [cot_agent_base(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc4 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4", "thinking of subtask_5", "answer of subtask_5"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3a, answer3a, thinking3b, answer3b], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, generating IUPAC names for products, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    if possible_answers_4.count(answer4_content) < 2:
        debate_agents_4 = [cot_agent_base(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
        N_debate_rounds = N_round
        all_thinking4 = [[] for _ in range(N_debate_rounds)]
        all_answer4 = [[] for _ in range(N_debate_rounds)]
        for r in range(N_debate_rounds):
            for i, agent in enumerate(debate_agents_4):
                if r == 0:
                    input_infos_4 = [taskInfo, thinking3a, answer3a, thinking3b, answer3b] + possible_answers_4
                else:
                    input_infos_4 = [taskInfo, thinking3a, answer3a, thinking3b, answer3b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking_d, answer_d = await agent(input_infos_4, cot_instruction_4, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, debating IUPAC naming, thinking: {thinking_d.content}; answer: {answer_d.content}")
                all_thinking4[r].append(thinking_d)
                all_answer4[r].append(answer_d)
        final_decision_agent_4 = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
        thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 6: Select the most accurate IUPAC names for products A and B after debate.", is_sub_task=True)
        agents.append(f"Final Decision agent, selecting IUPAC names, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 6: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 7: Compare the predicted and named products from Subtask 6 against each multiple-choice option. Perform atom-level connectivity and stereochemical matching rather than relying solely on name strings to identify the correct match for both reactions A and B."
    debate_agents_5 = [cot_agent_base(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    all_thinking5 = [[] for _ in range(N_round)]
    all_answer5 = [[] for _ in range(N_round)]
    subtask_desc5 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_6", "answer of subtask_6"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                input_infos_5 = [taskInfo, thinking4, answer4]
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
            thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted products with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = cot_agent_base(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 8: Select and output the multiple-choice answer (A, B, C, or D) that correctly corresponds to the predicted products for reactions A and B based on the comprehensive structural and naming comparison.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct multiple-choice answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs