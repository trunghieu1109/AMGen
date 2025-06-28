async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the Michael reaction mechanism focusing specifically on the given substrates, emphasizing nucleophilic addition to α,β-unsaturated carbonyl compounds, detailing regioselectivity, stereochemistry, and possible enolate tautomers relevant to the reactions."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing Michael reaction mechanism with substrate focus, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    N = self.max_sc
    cot_sc_instruction_2a = "Sub-task 2a: Convert the IUPAC name 'methyl 2-oxocyclohexane-1-carboxylate' into an explicit structural representation including atom numbering, bond connectivity, and identification of reactive sites relevant to the Michael reaction, based on Sub-task 1 analysis."
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, converting IUPAC name methyl 2-oxocyclohexane-1-carboxylate, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_sc_instruction_2b = "Sub-task 2b: Convert the IUPAC name '2,4-dimethyl-1-(vinylsulfinyl)benzene' into an explicit structural representation including atom numbering, bond connectivity, and identification of nucleophilic or electrophilic sites relevant to the Michael reaction, based on Sub-task 1 analysis."
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, converting IUPAC name 2,4-dimethyl-1-(vinylsulfinyl)benzene, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_sc_instruction_3a = "Sub-task 3a: Convert the IUPAC name 'ethyl 2-ethylbutanoate' into an explicit structural representation including atom numbering, bond connectivity, and identification of reactive sites relevant to the Michael reaction, based on Sub-task 1 analysis."
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, converting IUPAC name ethyl 2-ethylbutanoate, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    answer3a_content = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[answer3a_content]
    answer3a = answermapping_3a[answer3a_content]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_sc_instruction_3b = "Sub-task 3b: Convert the IUPAC name 'methyl 2-cyclopentylidene-2-phenylacetate' into an explicit structural representation including atom numbering, bond connectivity, and clarification of the location and conjugation of the double bond to avoid ambiguity, based on Sub-task 1 analysis."
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, converting IUPAC name methyl 2-cyclopentylidene-2-phenylacetate, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    answer3b_content = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[answer3b_content]
    answer3b = answermapping_3b[answer3b_content]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_sc_instruction_4a = "Sub-task 4a: Generate multiple plausible Michael reaction product structures for reaction A by applying the mechanism to substrates from subtasks 2a and 2b, considering all relevant regioisomers, stereoisomers, and enolate tautomers; document each candidate with detailed structural rationale."
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.7) for _ in range(N)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking2a, answer2a, thinking2b, answer2b], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, generating plausible products for reaction A, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_sc_instruction_4b = "Sub-task 4b: Generate multiple plausible Michael reaction product structures for reaction B by applying the mechanism to substrates from subtasks 3a and 3b, considering all relevant regioisomers, stereoisomers, and enolate tautomers; document each candidate with detailed structural rationale."
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.7) for _ in range(N)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking3a, answer3a, thinking3b, answer3b], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, generating plausible products for reaction B, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_reflect_instruction_5a = "Sub-task 5a: Critically evaluate the candidate products from Sub-task 4a using chemical principles, stereochemical and regioselective considerations, and literature precedents to select the most chemically plausible product for reaction A."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_5a = [taskInfo, thinking4a, answer4a]
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_reflect_instruction_5a,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Reflexion"
    }
    thinking5a, answer5a = await cot_agent_5a(cot_inputs_5a, cot_reflect_instruction_5a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5a.id}, evaluating products for reaction A, thinking: {thinking5a.content}; answer: {answer5a.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_5a([taskInfo, thinking5a, answer5a], "please review the chemical plausibility, stereochemical and regioselective correctness of the selected product for reaction A.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5a.id}, providing feedback on reaction A product evaluation, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5a.extend([thinking5a, answer5a, feedback])
        thinking5a, answer5a = await cot_agent_5a(cot_inputs_5a, cot_reflect_instruction_5a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5a.id}, refining evaluation for reaction A, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_reflect_instruction_5b = "Sub-task 5b: Critically evaluate the candidate products from Sub-task 4b using chemical principles, stereochemical and regioselective considerations, and literature precedents to select the most chemically plausible product for reaction B; include a reflexion step to verify substrate structure interpretation and product consistency."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5b = [taskInfo, thinking4b, answer4b, thinking3b, answer3b]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_reflect_instruction_5b,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_reflect_instruction_5b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, evaluating products for reaction B, thinking: {thinking5b.content}; answer: {answer5b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_5b([taskInfo, thinking5b, answer5b], "please review the substrate structure interpretation and product consistency for reaction B.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5b.id}, providing feedback on reaction B product evaluation, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5b.extend([thinking5b, answer5b, feedback])
        thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_reflect_instruction_5b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, refining evaluation for reaction B, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    debate_instruction_6 = "Sub-task 6: Compare the selected products from Sub-tasks 5a and 5b with the given multiple-choice options, analyzing structural features, nomenclature, and stereochemical details to determine the correct matches for A and B."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5a, answer5a, thinking5b, answer5b], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5a, answer5a, thinking5b, answer5b] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing selected products with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    sub_tasks.append(f"Sub-task 6 output: thinking - {all_thinking6[-1]}; answer - {all_answer6[-1]}")
    subtask_desc6['response'] = {"thinking": all_thinking6[-1], "answer": all_answer6[-1]}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 7: Select the correct multiple-choice answer (A, B, C, or D) based on the comparison in Sub-task 6 and provide the final answer formatted as per the query requirements.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct multiple-choice answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": "Sub-task 7: Select the correct multiple-choice answer based on comparison.",
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs