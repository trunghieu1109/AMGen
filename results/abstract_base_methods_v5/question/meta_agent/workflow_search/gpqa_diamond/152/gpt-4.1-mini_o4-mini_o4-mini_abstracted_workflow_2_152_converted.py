async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1: Analyze the reactants and reaction conditions for Michael addition reaction (A): dimethyl malonate + methyl (E)-3-(p-tolyl)acrylate + (NaOEt, EtOH). Identify the nucleophile, electrophile, and mechanistic roles of each component to understand the reaction pathway."
    cot_instruction_1b = "Sub-task 2: Analyze the reactants and reaction conditions for Michael addition reaction (B): 1-(cyclohex-1-en-1-yl)piperidine + (E)-but-2-enenitrile + (MeOH, H3O+). Identify the nucleophile, electrophile, and mechanistic roles of each component to understand the reaction pathway."
    cot_instruction_1c = "Sub-task 3: Analyze the reactants and reaction conditions for Michael addition reaction (C): compound C + but-3-en-2-one + (KOH, H2O). Identify the nucleophile, electrophile, and mechanistic roles of each component to understand the reaction pathway."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, analyzing reaction (A), thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1: ", sub_tasks[-1])
    subtask_desc_1b = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1b,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, analyzing reaction (B), thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 2: ", sub_tasks[-1])
    subtask_desc_1c = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_1c,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1c, answer1c = await cot_agent_1c([taskInfo], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, analyzing reaction (C), thinking: {thinking1c.content}; answer: {answer1c.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking1c, "answer": answer1c}
    logs.append(subtask_desc_1c)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction_2a = "Sub-task 4: Determine the major resonance-stabilized intermediate formed after nucleophilic attack at the β-position in reaction (A), based on the Michael addition mechanism and the identified reactants and conditions."
    cot_sc_instruction_2b = "Sub-task 5: Determine the major resonance-stabilized intermediate formed after nucleophilic attack at the β-position in reaction (B), based on the Michael addition mechanism and the identified reactants and conditions."
    cot_sc_instruction_2c = "Sub-task 6: Determine the major resonance-stabilized intermediate formed after nucleophilic attack at the β-position in reaction (C), based on the Michael addition mechanism and the identified reactants and conditions."
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_2c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc_2a = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1a, answer1a], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, determining intermediate for reaction (A), thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc_2a)
    print("Step 4: ", sub_tasks[-1])
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking1b, answer1b], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, determining intermediate for reaction (B), thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc_2b)
    print("Step 5: ", sub_tasks[-1])
    possible_answers_2c = []
    thinkingmapping_2c = {}
    answermapping_2c = {}
    subtask_desc_2c = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_2c,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2c, answer2c = await cot_agents_2c[i]([taskInfo, thinking1c, answer1c], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2c[i].id}, determining intermediate for reaction (C), thinking: {thinking2c.content}; answer: {answer2c.content}")
        possible_answers_2c.append(answer2c.content)
        thinkingmapping_2c[answer2c.content] = thinking2c
        answermapping_2c[answer2c.content] = answer2c
    answer2c_content = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking2c = thinkingmapping_2c[answer2c_content]
    answer2c = answermapping_2c[answer2c_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking2c, "answer": answer2c}
    logs.append(subtask_desc_2c)
    print("Step 6: ", sub_tasks[-1])
    cot_reflect_instruction_3a = "Sub-task 7: Predict the major final product for reaction (A) by completing the Michael addition reaction, considering reaction conditions, possible tautomerizations or protonations. Provide both the IUPAC name with correct numbering and a detailed skeletal connectivity description (e.g., SMILES or explicit carbon skeleton) to enable nomenclature validation."
    cot_reflect_instruction_3b = "Sub-task 8: Predict the major final product for reaction (B) by completing the Michael addition reaction, considering reaction conditions, possible tautomerizations or protonations. Provide both the IUPAC name with correct numbering and a detailed skeletal connectivity description to enable nomenclature validation."
    cot_reflect_instruction_3c = "Sub-task 9: Predict the major final product for reaction (C) by completing the Michael addition reaction, considering reaction conditions, possible tautomerizations or protonations. Provide both the IUPAC name with correct numbering and a detailed skeletal connectivity description to enable nomenclature validation."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3a = [taskInfo, thinking2a, answer2a]
    cot_inputs_3b = [taskInfo, thinking2b, answer2b]
    cot_inputs_3c = [taskInfo, thinking2c, answer2c]
    thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, predicting final product for reaction (A), thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3a([taskInfo, thinking3a, answer3a], "Review the predicted final product for reaction (A) and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, feedback on reaction (A), thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback])
        thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining final product for reaction (A), thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc_3a = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc_3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc_3a)
    print("Step 7: ", sub_tasks[-1])
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, predicting final product for reaction (B), thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "Review the predicted final product for reaction (B) and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, feedback on reaction (B), thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining final product for reaction (B), thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc_3b = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc_3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc_3b)
    print("Step 8: ", sub_tasks[-1])
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, predicting final product for reaction (C), thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3c([taskInfo, thinking3c, answer3c], "Review the predicted final product for reaction (C) and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, feedback on reaction (C), thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining final product for reaction (C), thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc_3c = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc_3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc_3c)
    print("Step 9: ", sub_tasks[-1])
    nomenclature_validation_instruction = "Sub-task 10: Perform nomenclature validation for the predicted major final products of reactions (A), (B), and (C). Verify the correctness of IUPAC names, numbering, and substituent positions by cross-checking with skeletal connectivity descriptions. Identify and correct any naming errors to prevent propagation. Provide corrected names and detailed skeletal connectivity descriptions."
    cot_agents_10 = [LLMAgentBase(["thinking", "answer"], "Self-Consistency Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_10 = []
    thinkingmapping_10 = {}
    answermapping_10 = {}
    subtask_desc_10 = {
        "subtask_id": "subtask_10",
        "instruction": nomenclature_validation_instruction,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7", "thinking of subtask 8", "answer of subtask 8", "thinking of subtask 9", "answer of subtask 9"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking10, answer10 = await cot_agents_10[i]([taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c], nomenclature_validation_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_10[i].id}, validating nomenclature, thinking: {thinking10.content}; answer: {answer10.content}")
        possible_answers_10.append(answer10.content)
        thinkingmapping_10[answer10.content] = thinking10
        answermapping_10[answer10.content] = answer10
    answer10_content = Counter(possible_answers_10).most_common(1)[0][0]
    thinking10 = thinkingmapping_10[answer10_content]
    answer10 = answermapping_10[answer10_content]
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc_10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc_10)
    print("Step 10: ", sub_tasks[-1])
    list_confirm_instruction = "Sub-task 11: Explicitly list and confirm the final predicted major products for reactions (A), (B), and (C) with validated chemical names and structural descriptions, ensuring clarity and unambiguity."
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_11 = {
        "subtask_id": "subtask_11",
        "instruction": list_confirm_instruction,
        "context": ["user query", "thinking of subtask 10", "answer of subtask 10"],
        "agent_collaboration": "CoT"
    }
    thinking11, answer11 = await cot_agent_11([taskInfo, thinking10, answer10], list_confirm_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_11.id}, confirming final products, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc_11['response'] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc_11)
    print("Step 11: ", sub_tasks[-1])
    mapping_instruction = "Sub-task 12: Map the validated predicted products from Sub-task 11 to the multiple-choice options provided in the query. Clarify whether each label (A, B, C) corresponds to reactants or products, resolve any ambiguities in labeling (especially for compound C), and identify the correct matching choice."
    debate_agents_12 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_12 = self.max_round
    all_thinking_12 = [[] for _ in range(N_max_12)]
    all_answer_12 = [[] for _ in range(N_max_12)]
    subtask_desc_12 = {
        "subtask_id": "subtask_12",
        "instruction": mapping_instruction,
        "context": ["user query", "thinking of subtask 11", "answer of subtask 11"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_12):
        for i, agent in enumerate(debate_agents_12):
            if r == 0:
                thinking12, answer12 = await agent([taskInfo, thinking11, answer11], mapping_instruction, r, is_sub_task=True)
            else:
                input_infos_12 = [taskInfo, thinking11, answer11] + all_thinking_12[r-1] + all_answer_12[r-1]
                thinking12, answer12 = await agent(input_infos_12, mapping_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mapping products to choices, thinking: {thinking12.content}; answer: {answer12.content}")
            all_thinking_12[r].append(thinking12)
            all_answer_12[r].append(answer12)
    final_decision_agent_12 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking12, answer12 = await final_decision_agent_12([taskInfo] + all_thinking_12[-1] + all_answer_12[-1], "Sub-task 12: Make final decision on the correct matching choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct choice, thinking: {thinking12.content}; answer: {answer12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    subtask_desc_12['response'] = {"thinking": thinking12, "answer": answer12}
    logs.append(subtask_desc_12)
    print("Step 12: ", sub_tasks[-1])
    final_validation_instruction = "Sub-task 13: Conduct a final validation by cross-checking the predicted products and the selected multiple-choice answer against the original question wording and chemical logic to ensure consistency, correctness, and completeness of the final answer."
    cot_agent_13 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_13 = {
        "subtask_id": "subtask_13",
        "instruction": final_validation_instruction,
        "context": ["user query", "thinking of subtask 12", "answer of subtask 12"],
        "agent_collaboration": "CoT"
    }
    thinking13, answer13 = await cot_agent_13([taskInfo, thinking12, answer12], final_validation_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_13.id}, final validation, thinking: {thinking13.content}; answer: {answer13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking13.content}; answer - {answer13.content}")
    subtask_desc_13['response'] = {"thinking": thinking13, "answer": answer13}
    logs.append(subtask_desc_13)
    print("Step 13: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking13, answer13, sub_tasks, agents)
    return final_answer, logs