async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the Michael reaction mechanism in detail, focusing on nucleophilic addition to alpha,beta-unsaturated carbonyl compounds, including regiochemistry, stereochemistry, and typical electrophilic sites, to establish a mechanistic framework for product prediction."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzed Michael reaction mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    N = self.max_sc
    cot_sc_instruction_2 = "Sub-task 2: Characterize the reactants in reaction A (methyl 2-oxocyclohexane-1-carboxylate and 2,4-dimethyl-1-(vinylsulfinyl)benzene with NaOEt in THF) by identifying all relevant functional groups, reactive sites, and provide explicit structural representations (e.g., SMILES or skeletal formulas) to clarify electrophilic and nucleophilic centers, based on Sub-task 1 output."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, characterized reactants in reaction A, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_sc_instruction_3 = "Sub-task 3: Characterize the reactants in reaction B (ethyl 2-ethylbutanoate and methyl 2-cyclopentylidene-2-phenylacetate with NaH in THF) by identifying all relevant functional groups, reactive sites, and provide explicit structural representations (e.g., SMILES or skeletal formulas) to clarify electrophilic and nucleophilic centers, especially focusing on the alpha,beta-unsaturated ester moiety, based on Sub-task 1 output."
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, characterized reactants in reaction B, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_reflect_instruction_4 = "Sub-task 4: Predict the product structure of reaction A by applying the Michael addition mechanism to the characterized reactants, determining the precise site of nucleophilic attack, new carbon-carbon bond formation, and resulting stereochemistry and functional groups, using clear atom mapping and precise chemical nomenclature, based on Sub-task 2 output."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_4 = [taskInfo, thinking2, answer2]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, predicted product structure of reaction A, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4], "Critically evaluate the predicted product structure of reaction A for correctness, atom mapping, stereochemistry, and nomenclature precision.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback on reaction A product prediction, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refined product prediction of reaction A, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_sc_instruction_5a = "Sub-task 5a: Enumerate all possible electrophilic centers on the alpha,beta-unsaturated ester methyl 2-cyclopentylidene-2-phenylacetate, using provided structural representations to avoid ambiguity, and identify potential Michael acceptor sites, based on Sub-task 3 output."
    cot_agents_5a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5a = []
    thinkingmapping_5a = {}
    answermapping_5a = {}
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_sc_instruction_5a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5a, answer5a = await cot_agents_5a[i]([taskInfo, thinking3, answer3], cot_sc_instruction_5a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5a[i].id}, enumerated electrophilic centers on methyl 2-cyclopentylidene-2-phenylacetate, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a.content)
        thinkingmapping_5a[answer5a.content] = thinking5a
        answermapping_5a[answer5a.content] = answer5a
    answer5a_content = Counter(possible_answers_5a).most_common(1)[0][0]
    thinking5a = thinkingmapping_5a[answer5a_content]
    answer5a = answermapping_5a[answer5a_content]
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a["response"] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_sc_instruction_5b = "Sub-task 5b: Evaluate and select the correct Michael addition site on methyl 2-cyclopentylidene-2-phenylacetate based on conjugation, resonance stabilization, and steric factors, applying a Self-Consistency Chain-of-Thought approach to generate and compare multiple hypotheses, based on Sub-task 5a output."
    cot_agents_5b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5b = []
    thinkingmapping_5b = {}
    answermapping_5b = {}
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_sc_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5b, answer5b = await cot_agents_5b[i]([taskInfo, thinking5a, answer5a], cot_sc_instruction_5b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5b[i].id}, evaluated and selected Michael addition site, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers_5b.append(answer5b.content)
        thinkingmapping_5b[answer5b.content] = thinking5b
        answermapping_5b[answer5b.content] = answer5b
    answer5b_content = Counter(possible_answers_5b).most_common(1)[0][0]
    thinking5b = thinkingmapping_5b[answer5b_content]
    answer5b = answermapping_5b[answer5b_content]
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b["response"] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_reflect_instruction_5c = "Sub-task 5c: Predict the product structure of reaction B by applying the Michael addition mechanism to the selected electrophilic site and nucleophile, determining the new carbon-carbon bond formation, stereochemistry, and functional groups, with precise atom mapping and chemical nomenclature, based on Sub-task 5b output."
    cot_agent_5c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5c = [taskInfo, thinking5b, answer5b]
    thinking5c, answer5c = await cot_agent_5c(cot_inputs_5c, cot_reflect_instruction_5c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5c.id}, predicted product structure of reaction B, thinking: {thinking5c.content}; answer: {answer5c.content}")
    for i in range(N_max):
        feedback5c, correct5c = await critic_agent_5c([taskInfo, thinking5c, answer5c], "Critically evaluate the predicted product structure of reaction B for correctness, atom mapping, stereochemistry, and nomenclature precision. If errors or omissions are found, propose corrections.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5c.id}, feedback on reaction B product prediction, thinking: {feedback5c.content}; answer: {correct5c.content}")
        if correct5c.content == "True":
            break
        cot_inputs_5c.extend([thinking5c, answer5c, feedback5c])
        thinking5c, answer5c = await cot_agent_5c(cot_inputs_5c, cot_reflect_instruction_5c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5c.id}, refined product prediction of reaction B, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_reflect_instruction_5c,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc5c["response"] = {"thinking": thinking5c, "answer": answer5c}
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    debate_instruction_5d = "Sub-task 5d: Perform an internal verification and error correction loop on the predicted product of reaction B, where critic agents actively propose corrections if inconsistencies or misassignments are detected, triggering re-analysis and refinement of the product prediction before proceeding."
    debate_agents_5d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_5d = [[] for _ in range(N_max)]
    all_answer_5d = [[] for _ in range(N_max)]
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": debate_instruction_5d,
        "context": ["user query", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents_5d):
            if r == 0:
                thinking5d, answer5d = await agent([taskInfo, thinking5c, answer5c], debate_instruction_5d, r, is_sub_task=True)
            else:
                input_infos_5d = [taskInfo, thinking5c, answer5c] + all_thinking_5d[r-1] + all_answer_5d[r-1]
                thinking5d, answer5d = await agent(input_infos_5d, debate_instruction_5d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying and correcting product B prediction, thinking: {thinking5d.content}; answer: {answer5d.content}")
            all_thinking_5d[r].append(thinking5d)
            all_answer_5d[r].append(answer5d)
    final_decision_agent_5d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5d, answer5d = await final_decision_agent_5d([taskInfo] + all_thinking_5d[-1] + all_answer_5d[-1], "Sub-task 5d: Make final decision on the verified and corrected product structure of reaction B.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding verified product B, thinking: {thinking5d.content}; answer: {answer5d.content}")
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d["response"] = {"thinking": thinking5d, "answer": answer5d}
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])
    
    debate_instruction_6 = "Sub-task 6: Compare the predicted product structures of reactions A and B with each of the provided multiple-choice options, analyzing the consistency of substituents, positions, stereochemistry, and nomenclature to identify the correct match, incorporating a targeted Debate between agents proposing alternative interpretations for ambiguous features."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = [[] for _ in range(N_max)]
    all_answer6 = [[] for _ in range(N_max)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5d", "answer of subtask 5d"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4, answer4, thinking5d, answer5d], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking4, answer4, thinking5d, answer5d] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted products with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct multiple-choice option matching predicted products.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_reflect_instruction_7 = "Sub-task 7: Conduct a final verification step to re-validate the connectivity, substitution patterns, and stereochemical assignments of both predicted products against the original reactants and reaction conditions to ensure consistency and correctness before final answer selection."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_7 = [taskInfo, thinking4, answer4, thinking5d, answer5d, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, final verification of predicted products, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7, answer7], "Critically evaluate the final verification of predicted products for consistency and correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, feedback on final verification, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refined final verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5d", "answer of subtask 5d", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Sub-task 8: Select and return the letter (A, B, C, or D) corresponding to the choice that correctly matches the verified predicted products for both reactions A and B, based on Sub-task 7 output."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"Final selection agent {cot_agent_8.id}, selected correct multiple-choice answer, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    subtask_desc8["response"] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs