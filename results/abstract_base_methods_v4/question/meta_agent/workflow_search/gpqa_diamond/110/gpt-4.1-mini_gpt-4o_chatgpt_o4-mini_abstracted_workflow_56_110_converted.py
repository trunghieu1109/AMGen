async def forward_110(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_1a = (
        "Sub-task 1a: Map and number all carbon and relevant heteroatoms in 2-ethyl-2,6-dimethylcyclohexan-1-one and ethyl acrylate. "
        "Identify all enolizable alpha-carbons on the cyclohexanone ring with explicit atom numbering, considering ring geometry and substituent positions. "
        "Provide detailed atom labels and a mini-table mapping atoms."
    )
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_sc_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}
    for i in range(self.max_sc):
        thinking_1a, answer_1a = await cot_sc_agents_1a[i]([taskInfo], cot_sc_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1a[i].id}, mapping and numbering atoms in reactants, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
        possible_answers_1a.append(answer_1a.content)
        thinkingmapping_1a[answer_1a.content] = thinking_1a
        answermapping_1a[answer_1a.content] = answer_1a
    most_common_answer_1a = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking_1a = thinkingmapping_1a[most_common_answer_1a]
    answer_1a = answermapping_1a[most_common_answer_1a]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_sc_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_1b = (
        "Sub-task 1b: Analyze the enolate formation regioselectivity from 2-ethyl-2,6-dimethylcyclohexan-1-one under t-BuOK conditions. "
        "Consider acidity (pKa), steric hindrance, and ring conformations. Determine the predominant enolate species formed with explicit atom labels. "
        "Use atom numbering from Sub-task 1a."
    )
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    for i in range(self.max_sc):
        thinking_1b, answer_1b = await cot_sc_agents_1b[i]([taskInfo, thinking_1a, answer_1a], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1b[i].id}, analyzing enolate regioselectivity, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
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
    
    cot_sc_agents_1c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_1c = (
        "Sub-task 1c: Examine the Michael addition mechanism between the identified enolate and ethyl acrylate. "
        "Map nucleophilic attack site, regiochemistry, and stereochemical outcomes. Consider ring geometry, steric hindrance, and possible transition states. "
        "Predict the major adduct structure with numbered atoms and provide atom mapping from reactants to product."
    )
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_sc_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    for i in range(self.max_sc):
        thinking_1c, answer_1c = await cot_sc_agents_1c[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1c[i].id}, examining Michael addition mechanism, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
        possible_answers_1c.append(answer_1c.content)
        thinkingmapping_1c[answer_1c.content] = thinking_1c
        answermapping_1c[answer_1c.content] = answer_1c
    most_common_answer_1c = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking_1c = thinkingmapping_1c[most_common_answer_1c]
    answer_1c = answermapping_1c[most_common_answer_1c]
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_sc_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_2a = (
        "Sub-task 2a: Map and number all carbon and heteroatoms in 1-nitropropane, (E)-but-2-enenitrile, KOH, and H2O. "
        "Explicitly label atoms to track throughout the reaction mechanism. Provide detailed atom numbering and mapping tables."
    )
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    for i in range(self.max_sc):
        thinking_2a, answer_2a = await cot_sc_agents_2a[i]([taskInfo], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2a[i].id}, mapping and numbering atoms in second reaction reactants, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
        possible_answers_2a.append(answer_2a.content)
        thinkingmapping_2a[answer_2a.content] = thinking_2a
        answermapping_2a[answer_2a.content] = answer_2a
    most_common_answer_2a = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking_2a = thinkingmapping_2a[most_common_answer_2a]
    answer_2a = answermapping_2a[most_common_answer_2a]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_sc_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_2b = (
        "Sub-task 2b: Distinguish and analyze possible reaction pathways between 1-nitropropane and (E)-but-2-enenitrile under KOH/H2O conditions. "
        "Differentiate nitro-Michael addition from Henry-type (nitroaldol) mechanisms. Evaluate which pathway is favored based on acidity, nucleophilicity, and reaction conditions. "
        "Use atom numbering from Sub-task 2a."
    )
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    for i in range(self.max_sc):
        thinking_2b, answer_2b = await cot_sc_agents_2b[i]([taskInfo, thinking_2a, answer_2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2b[i].id}, analyzing reaction pathways, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
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
    
    cot_sc_agents_2c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_2c = (
        "Sub-task 2c: Construct the detailed product structure(s) from the favored reaction pathway(s). "
        "Ensure correct carbon count, regiochemistry, and stereochemistry. Provide explicit atom numbering and map reactant atoms to product atoms to verify consistency."
    )
    subtask_desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_sc_instruction_2c,
        "context": ["user query", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2c = []
    thinkingmapping_2c = {}
    answermapping_2c = {}
    for i in range(self.max_sc):
        thinking_2c, answer_2c = await cot_sc_agents_2c[i]([taskInfo, thinking_2b, answer_2b], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2c[i].id}, constructing product structure, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
        possible_answers_2c.append(answer_2c.content)
        thinkingmapping_2c[answer_2c.content] = thinking_2c
        answermapping_2c[answer_2c.content] = answer_2c
    most_common_answer_2c = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking_2c = thinkingmapping_2c[most_common_answer_2c]
    answer_2c = answermapping_2c[most_common_answer_2c]
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 2c: ", sub_tasks[-1])
    
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3a = self.max_round
    cot_inputs_3a = [taskInfo, thinking_1c, answer_1c]
    cot_reflect_instruction_3a = (
        "Sub-task 3a: Evaluate steric hindrance and electronic effects on all plausible isomers and stereoisomers of the product from the first reaction (Michael addition product). "
        "Compare transition states and intermediate stabilities to identify the major product isomer."
    )
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3a, answer_3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, evaluating steric and electronic effects for first reaction, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    for i in range(N_max_3a):
        feedback, correct = await critic_agent_3a([taskInfo, thinking_3a, answer_3a],
                                                 "please review the steric and electronic evaluation for the first reaction and provide its limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking_3a, answer_3a, feedback])
        thinking_3a, answer_3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining evaluation for first reaction, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking_3a, answer_3a]
    cot_reflect_instruction_3b = (
        "Sub-task 3b: Perform a self-consistency check on the first reaction product: verify atom counts, substituent positions, and stereochemical assignments. "
        "If inconsistencies are found, trigger reanalysis of subtasks 1b and 1c."
    )
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask_3a", "answer of subtask_3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3b, answer_3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, performing self-consistency check for first reaction product, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    for i in range(N_max_3b):
        feedback, correct = await critic_agent_3b([taskInfo, thinking_3b, answer_3b],
                                                 "please review the self-consistency check for the first reaction product and provide its limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking_3b, answer_3b, feedback])
        thinking_3b, answer_3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining self-consistency check for first reaction product, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4a = self.max_round
    cot_inputs_4a = [taskInfo, thinking_2c, answer_2c]
    cot_reflect_instruction_4a = (
        "Sub-task 4a: Evaluate steric and electronic factors influencing the formation of possible isomers from the second reaction product(s). "
        "Analyze kinetic versus thermodynamic control and transition state energies to select the major product isomer."
    )
    subtask_desc_4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_reflect_instruction_4a,
        "context": ["user query", "thinking of subtask_2c", "answer of subtask_2c"],
        "agent_collaboration": "Reflexion"
    }
    thinking_4a, answer_4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, evaluating steric and electronic effects for second reaction, thinking: {thinking_4a.content}; answer: {answer_4a.content}")
    for i in range(N_max_4a):
        feedback, correct = await critic_agent_4a([taskInfo, thinking_4a, answer_4a],
                                                 "please review the steric and electronic evaluation for the second reaction and provide its limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4a.extend([thinking_4a, answer_4a, feedback])
        thinking_4a, answer_4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, refining evaluation for second reaction, thinking: {thinking_4a.content}; answer: {answer_4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking_4a.content}; answer - {answer_4a.content}")
    subtask_desc_4a['response'] = {"thinking": thinking_4a, "answer": answer_4a}
    logs.append(subtask_desc_4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4b = self.max_round
    cot_inputs_4b = [taskInfo, thinking_4a, answer_4a]
    cot_reflect_instruction_4b = (
        "Sub-task 4b: Perform a self-consistency check on the second reaction product: verify carbon skeleton, regiochemistry, and stereochemistry. "
        "If discrepancies arise, prompt re-examination of subtasks 2b and 2c."
    )
    subtask_desc_4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_reflect_instruction_4b,
        "context": ["user query", "thinking of subtask_4a", "answer of subtask_4a"],
        "agent_collaboration": "Reflexion"
    }
    thinking_4b, answer_4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, performing self-consistency check for second reaction product, thinking: {thinking_4b.content}; answer: {answer_4b.content}")
    for i in range(N_max_4b):
        feedback, correct = await critic_agent_4b([taskInfo, thinking_4b, answer_4b],
                                                 "please review the self-consistency check for the second reaction product and provide its limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4b.extend([thinking_4b, answer_4b, feedback])
        thinking_4b, answer_4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining self-consistency check for second reaction product, thinking: {thinking_4b.content}; answer: {answer_4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking_4b.content}; answer - {answer_4b.content}")
    subtask_desc_4b['response'] = {"thinking": thinking_4b, "answer": answer_4b}
    logs.append(subtask_desc_4b)
    print("Step 4b: ", sub_tasks[-1])
    
    debate_instruction_5 = (
        "Sub-task 5: Critically compare the major products identified from both reactions with the given multiple-choice options. "
        "Cross-validate atom numbering, substituent positions, and product names. Employ adversarial review to challenge mechanistic assumptions and confirm the correct choice (A, B, C, or D)."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_3b", "answer of subtask_3b", "thinking of subtask_4b", "answer of subtask_4b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_3b, answer_3b, thinking_4b, answer_4b], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_3b, answer_3b, thinking_4b, answer_4b] + all_thinking5[r-1] + all_answer5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, critically comparing products and selecting correct choice, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking5[r].append(thinking_5)
            all_answer5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs