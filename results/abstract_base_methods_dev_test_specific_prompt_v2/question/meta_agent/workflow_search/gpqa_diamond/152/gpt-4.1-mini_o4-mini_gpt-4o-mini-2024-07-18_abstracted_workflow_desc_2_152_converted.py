async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1: Analyze reaction (A) reactants and conditions: dimethyl malonate + methyl (E)-3-(p-tolyl)acrylate + (NaOEt, EtOH). Identify the nucleophile, electrophile, site of nucleophilic attack, and expected Michael addition mechanism steps."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, analyzing reaction (A), thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 2: Analyze reaction (B) reactants and conditions: 1-(cyclohex-1-en-1-yl)piperidine + (E)-but-2-enenitrile + (MeOH, H3O+). Identify the nucleophile, electrophile, site of nucleophilic attack, and expected Michael addition mechanism steps including protonation."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1b,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, analyzing reaction (B), thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_1c = "Sub-task 3: Analyze reaction (C) reactants and conditions: C + but-3-en-2-one + (KOH, H2O) with product 2-(3-oxobutyl)cyclohexane-1,3-dione. Deduce identity of C, nucleophile, electrophile, and Michael addition mechanism steps."
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1c = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_1c,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1c, answer_1c = await cot_agent_1c([taskInfo], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, analyzing reaction (C), thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_2a = "Sub-task 4: For reaction (A), predict the major product structure by applying Michael addition mechanism: nucleophile attack at β-carbon, followed by tautomerization and rearrangement if any. Explicitly determine the IUPAC numbering prioritizing carboxylate groups and assign substituent positions accordingly."
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
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
        thinking_i, answer_i = await cot_agents_2a[i]([taskInfo, thinking_1a, answer_1a], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, predicting product (A), thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2a.append(answer_i.content)
        thinkingmapping_2a[answer_i.content] = thinking_i
        answermapping_2a[answer_i.content] = answer_i
    answer_2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking_2a = thinkingmapping_2a[answer_2a_content]
    answer_2a = answermapping_2a[answer_2a_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 4: ", sub_tasks[-1])
    
    cot_sc_instruction_2b = "Sub-task 5: For reaction (B), predict the major product structure by applying Michael addition mechanism: nucleophile attack at β-carbon, followed by protonation and tautomerization under acidic methanol conditions. Explicitly determine the IUPAC numbering and assign substituent positions accordingly."
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
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
        thinking_i, answer_i = await cot_agents_2b[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, predicting product (B), thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2b.append(answer_i.content)
        thinkingmapping_2b[answer_i.content] = thinking_i
        answermapping_2b[answer_i.content] = answer_i
    answer_2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[answer_2b_content]
    answer_2b = answermapping_2b[answer_2b_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 5: ", sub_tasks[-1])
    
    cot_sc_instruction_2c = "Sub-task 6: For reaction (C), predict the major product structure by applying Michael addition mechanism: nucleophile attack at β-carbon, followed by tautomerization under basic aqueous conditions. Explicitly determine the IUPAC numbering and assign substituent positions accordingly."
    cot_agents_2c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
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
        thinking_i, answer_i = await cot_agents_2c[i]([taskInfo, thinking_1c, answer_1c], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2c[i].id}, predicting product (C), thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2c.append(answer_i.content)
        thinkingmapping_2c[answer_i.content] = thinking_i
        answermapping_2c[answer_i.content] = answer_i
    answer_2c_content = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking_2c = thinkingmapping_2c[answer_2c_content]
    answer_2c = answermapping_2c[answer_2c_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 6: ", sub_tasks[-1])
    
    reflexion_instruction_3a = "Sub-task 7: Normalize predicted products from subtasks 4, 5, and 6 to their major tautomeric forms (keto vs enol) based on reaction conditions and standard chemical knowledge."
    reflexion_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3a = self.max_round
    cot_inputs_3a = [taskInfo, thinking_2a, answer_2a, thinking_2b, answer_2b, thinking_2c, answer_2c]
    subtask_desc_3a = {
        "subtask_id": "subtask_7",
        "instruction": reflexion_instruction_3a,
        "context": ["user query", "thinking and answer of subtasks 4,5,6"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3a, answer_3a = await reflexion_agent_3a(cot_inputs_3a, reflexion_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent_3a.id}, normalizing tautomeric forms, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    for i in range(N_max_3a):
        feedback, correct = await critic_agent_3a([taskInfo, thinking_3a, answer_3a], "Please review the tautomer normalization and provide feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking_3a, answer_3a, feedback])
        thinking_3a, answer_3a = await reflexion_agent_3a(cot_inputs_3a, reflexion_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {reflexion_agent_3a.id}, refining tautomer normalization, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_3b = "Sub-task 8: Convert the normalized product structures from subtask 7 into IUPAC names with correct numbering and substituent positions, ensuring consistency with carboxylate priority rules and tautomer preference."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3b = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_3b,
        "context": ["user query", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking_3b, answer_3b = await cot_agent_3b([taskInfo, answer_3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, converting products to IUPAC names, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 8: ", sub_tasks[-1])
    
    cot_instruction_3c = "Sub-task 9: Compare the IUPAC names generated in subtask 8 against all given answer choices (A, B, C, D) using pattern matching to identify the best matching choice for each product (A, B, C)."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3c = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_3c,
        "context": ["user query", "answer of subtask 8"],
        "agent_collaboration": "CoT"
    }
    thinking_3c, answer_3c = await cot_agent_3c([taskInfo, answer_3b], cot_instruction_3c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3c.id}, comparing IUPAC names to choices, thinking: {thinking_3c.content}; answer: {answer_3c.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking_3c.content}; answer - {answer_3c.content}")
    subtask_desc_3c['response'] = {"thinking": thinking_3c, "answer": answer_3c}
    logs.append(subtask_desc_3c)
    print("Step 9: ", sub_tasks[-1])
    
    comparator_instruction_3d = "Sub-task 10: Perform a final consistency check to ensure all three predicted products correspond to a single answer choice. Flag any mismatches or ambiguities for review."
    comparator_agent_3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3d = {
        "subtask_id": "subtask_10",
        "instruction": comparator_instruction_3d,
        "context": ["user query", "answer of subtask 9"],
        "agent_collaboration": "CoT"
    }
    thinking_3d, answer_3d = await comparator_agent_3d([taskInfo, answer_3c], comparator_instruction_3d, is_sub_task=True)
    agents.append(f"CoT agent {comparator_agent_3d.id}, final consistency check, thinking: {thinking_3d.content}; answer: {answer_3d.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking_3d.content}; answer - {answer_3d.content}")
    subtask_desc_3d['response'] = {"thinking": thinking_3d, "answer": answer_3d}
    logs.append(subtask_desc_3d)
    print("Step 10: ", sub_tasks[-1])
    
    debate_instruction_4 = "Sub-task 11: Debate among agents to finalize the best matching answer choice based on all previous subtasks' outputs and consistency checks."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_11",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking and answer of subtasks 10"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_3d, answer_3d], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_3d, answer_3d] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final choice, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 11: Make final decision on the best matching answer choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice decision, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 11: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs