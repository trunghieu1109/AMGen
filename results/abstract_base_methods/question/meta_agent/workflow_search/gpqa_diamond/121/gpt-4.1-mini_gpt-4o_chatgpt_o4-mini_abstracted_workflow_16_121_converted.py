async def forward_121(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Determine the chemical structure of product 1 formed by treating acetic acid with bromine, pyridine, and acetic anhydride under heating, based on known reaction mechanisms (Hell-Volhard-Zelinsky halogenation)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, determining product 1 structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Based on product 1 structure, determine the chemical structure of product 2 formed by heating product 1 with ethanol and a small amount of sulfuric acid, considering esterification of the acid and substitution reactions."
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining product 2 structure, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Sub-task 3: Based on outputs from sub-tasks 1 and 2, determine the chemical structure of product 3 formed by treating product 2 with sodium cyanide, identifying the site and nature of cyanide substitution or addition."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining product 3 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the product 3 structure determination and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining product 3 structure, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4a = "Sub-task 4a: Explore all plausible alkylation pathways of product 3 with excess sodium hydride and 1,5-dibromopentane, including mono- vs. di-alkylation and inter- vs. intramolecular reactions, to generate candidate structures for product 4. Use Self-Consistency Chain-of-Thought to consider multiple plausible mechanisms."
    N4a = self.max_sc
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4a)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking and answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4a):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, exploring alkylation pathways, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_sc_instruction_4b = "Sub-task 4b: Determine the correct ring size formed in product 4 by counting tether atoms and assessing ring closure feasibility, explicitly verifying that a six-membered ring (cyclohexane) is formed rather than a seven-membered ring. Use Self-Consistency Chain-of-Thought to evaluate multiple structural possibilities and select the most chemically reasonable."
    N4b = self.max_sc
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4b)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query", "thinking and answer of subtask 4a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4b):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking4a, answer4a], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, determining ring size, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_reflect_instruction_4c = "Sub-task 4c: Generate a clear, explicit structural representation of product 4 (e.g., SMILES, connectivity table, or numbered diagram) including ring size, substituent positions, and functional groups, to serve as input for NMR analysis. Verify the structure for clarity and correctness."
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4c = self.max_round
    cot_inputs_4c = [taskInfo, thinking4a, answer4a, thinking4b, answer4b]
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_reflect_instruction_4c,
        "context": ["user query", "thinking and answer of subtask 4a", "thinking and answer of subtask 4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, generating explicit structure, thinking: {thinking4c.content}; answer: {answer4c.content}")
    for i in range(N_max_4c):
        feedback, correct = await critic_agent_4c([taskInfo, thinking4c, answer4c], "please review the explicit structural representation of product 4 and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4c.extend([thinking4c, answer4c, feedback])
        thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_reflect_instruction_4c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, refining explicit structure, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {
        "thinking": thinking4c,
        "answer": answer4c
    }
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    
    cot_instruction_5a = "Sub-task 5a: Draw and number the final product 4's structure explicitly, based on subtask 4c output, to facilitate systematic identification of hydrogen environments."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking and answer of subtask 4c"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4c, answer4c], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, drawing and numbering structure, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_instruction_5b = "Sub-task 5b: Identify and list all chemically distinct hydrogen environments on substituents of product 4, including the ethyl ester group, cyano-substituted carbon, and any other side chains, based on the numbered structure from subtask 5a."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking and answer of subtask 5a"],
        "agent_collaboration": "CoT"
    }
    thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking5a, answer5a], cot_instruction_5b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5b.id}, identifying substituent hydrogens, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_instruction_5c = "Sub-task 5c: Analyze the ring hydrogens of product 4, considering stereochemistry, diastereotopic protons, and molecular symmetry to distinguish chemically distinct hydrogen signals, based on the numbered structure from subtask 5a."
    cot_agent_5c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_instruction_5c,
        "context": ["user query", "thinking and answer of subtask 5a"],
        "agent_collaboration": "CoT"
    }
    thinking5c, answer5c = await cot_agent_5c([taskInfo, thinking5a, answer5a], cot_instruction_5c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5c.id}, analyzing ring hydrogens, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {
        "thinking": thinking5c,
        "answer": answer5c
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    cot_instruction_5d = "Sub-task 5d: Sum all distinct hydrogen environments identified in subtasks 5b and 5c to determine the total number of chemically distinct hydrogen signals expected in the 1H NMR spectrum of product 4."
    cot_agent_5d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_instruction_5d,
        "context": ["user query", "thinking and answer of subtask 5b", "thinking and answer of subtask 5c"],
        "agent_collaboration": "CoT"
    }
    thinking5d, answer5d = await cot_agent_5d([taskInfo, thinking5b, answer5b, thinking5c, answer5c], cot_instruction_5d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5d.id}, summing distinct hydrogens, thinking: {thinking5d.content}; answer: {answer5d.content}")
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {
        "thinking": thinking5d,
        "answer": answer5d
    }
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])
    
    debate_instruction_6 = "Sub-task 6: Cross-validate the count of distinct hydrogen signals by independent enumeration and reflexion to ensure accuracy and consistency, then select the correct multiple-choice answer letter corresponding to the total count."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking and answer of subtask 5d"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5d, answer5d], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5d, answer5d] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validating hydrogen count, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the number of chemically distinct hydrogen signals and select the correct multiple-choice answer letter.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining distinct hydrogen signals and answer letter, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs