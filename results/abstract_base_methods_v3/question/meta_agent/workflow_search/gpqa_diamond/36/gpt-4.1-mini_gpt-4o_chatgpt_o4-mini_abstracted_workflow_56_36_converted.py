async def forward_36(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify and draw the chemical structure of intermediate A formed from the reaction of propionaldehyde with EDT and BF3, explicitly labeling all carbon atoms and functional groups to establish a clear starting structure."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying intermediate A, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    N = self.max_sc
    cot_sc_instruction_2 = "Sub-task 2: Determine the chemical structure of intermediate B formed by the reaction of A with BuLi, including the site of lithiation and updated carbon labeling to maintain structural clarity."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining intermediate B, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_sc_instruction_3 = "Sub-task 3: Determine the chemical structure of intermediate C formed by the reaction of B with bromoethane, specifying the substitution site and updating carbon labels to track changes precisely."
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, determining intermediate C, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4a = "Sub-task 4a: Write out the full molecular formula and detailed carbon skeleton of intermediate C, including explicit carbon numbering and substituent positions, to prepare for deprotection analysis."
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, writing molecular formula and carbon skeleton of intermediate C, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    most_common_answer_4a = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[most_common_answer_4a]
    answer4a = answermapping_4a[most_common_answer_4a]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_sc_instruction_4b = "Sub-task 4b: Perform the deprotection reaction of intermediate C with HgCl2, H2O, and H+, explicitly showing the mechanistic steps and resulting functional group transformations to obtain intermediate D."
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking4a, answer4a], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, performing deprotection to get intermediate D, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    most_common_answer_4b = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[most_common_answer_4b]
    answer4b = answermapping_4b[most_common_answer_4b]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_sc_instruction_4c = "Sub-task 4c: Confirm and verify the molecular formula, carbon skeleton, and structure of intermediate D, ensuring correct identification as pentan-3-one (not butan-2-one), with clear carbon numbering and substituent positions."
    cot_agents_4c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    possible_answers_4c = []
    thinkingmapping_4c = {}
    answermapping_4c = {}
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_sc_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4c, answer4c = await cot_agents_4c[i]([taskInfo, thinking4b, answer4b], cot_sc_instruction_4c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4c[i].id}, verifying intermediate D as pentan-3-one, thinking: {thinking4c.content}; answer: {answer4c.content}")
        possible_answers_4c.append(answer4c.content)
        thinkingmapping_4c[answer4c.content] = thinking4c
        answermapping_4c[answer4c.content] = answer4c
    most_common_answer_4c = Counter(possible_answers_4c).most_common(1)[0][0]
    thinking4c = thinkingmapping_4c[most_common_answer_4c]
    answer4c = answermapping_4c[most_common_answer_4c]
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    
    cot_sc_instruction_5a = "Sub-task 5a: Generate all plausible phosphonium ylides derived from 3-bromopentane reacting with PPh3 and BuLi, detailing their structures, carbon numbering, and possible resonance forms."
    cot_agents_5a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5a = []
    thinkingmapping_5a = {}
    answermapping_5a = {}
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_sc_instruction_5a,
        "context": ["user query", "thinking of subtask 4c", "answer of subtask 4c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5a, answer5a = await cot_agents_5a[i]([taskInfo, thinking4c, answer4c], cot_sc_instruction_5a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5a[i].id}, generating phosphonium ylides, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a.content)
        thinkingmapping_5a[answer5a.content] = thinking5a
        answermapping_5a[answer5a.content] = answer5a
    most_common_answer_5a = Counter(possible_answers_5a).most_common(1)[0][0]
    thinking5a = thinkingmapping_5a[most_common_answer_5a]
    answer5a = answermapping_5a[most_common_answer_5a]
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_sc_instruction_5b = "Sub-task 5b: Predict the Wittig reaction products formed by reacting intermediate D (pentan-3-one) with each plausible ylide, including stereochemistry, double bond position, and carbon skeleton for each candidate product."
    cot_agents_5b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5b = []
    thinkingmapping_5b = {}
    answermapping_5b = {}
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_sc_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a", "thinking of subtask 4c", "answer of subtask 4c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5b, answer5b = await cot_agents_5b[i]([taskInfo, thinking5a, answer5a, thinking4c, answer4c], cot_sc_instruction_5b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5b[i].id}, predicting Wittig products, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers_5b.append(answer5b.content)
        thinkingmapping_5b[answer5b.content] = thinking5b
        answermapping_5b[answer5b.content] = answer5b
    most_common_answer_5b = Counter(possible_answers_5b).most_common(1)[0][0]
    thinking5b = thinkingmapping_5b[most_common_answer_5b]
    answer5b = answermapping_5b[most_common_answer_5b]
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_sc_instruction_5c = "Sub-task 5c: Evaluate and compare the chemical plausibility and stability of each candidate Wittig product, considering steric and electronic factors, to select the most reasonable final product E with explicit structure and carbon labeling."
    cot_agents_5c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    possible_answers_5c = []
    thinkingmapping_5c = {}
    answermapping_5c = {}
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_sc_instruction_5c,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5c, answer5c = await cot_agents_5c[i]([taskInfo, thinking5b, answer5b], cot_sc_instruction_5c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5c[i].id}, evaluating Wittig products, thinking: {thinking5c.content}; answer: {answer5c.content}")
        possible_answers_5c.append(answer5c.content)
        thinkingmapping_5c[answer5c.content] = thinking5c
        answermapping_5c[answer5c.content] = answer5c
    most_common_answer_5c = Counter(possible_answers_5c).most_common(1)[0][0]
    thinking5c = thinkingmapping_5c[most_common_answer_5c]
    answer5c = answermapping_5c[most_common_answer_5c]
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {"thinking": thinking5c, "answer": answer5c}
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    cot_sc_instruction_5d = "Sub-task 5d: Verify the selected final product E by cross-checking carbon count, molecular formula, and consistency with all previous intermediates to ensure structural accuracy before NMR analysis."
    cot_agents_5d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    possible_answers_5d = []
    thinkingmapping_5d = {}
    answermapping_5d = {}
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_sc_instruction_5d,
        "context": ["user query", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5d, answer5d = await cot_agents_5d[i]([taskInfo, thinking5c, answer5c], cot_sc_instruction_5d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5d[i].id}, verifying final product E, thinking: {thinking5d.content}; answer: {answer5d.content}")
        possible_answers_5d.append(answer5d.content)
        thinkingmapping_5d[answer5d.content] = thinking5d
        answermapping_5d[answer5d.content] = answer5d
    most_common_answer_5d = Counter(possible_answers_5d).most_common(1)[0][0]
    thinking5d = thinkingmapping_5d[most_common_answer_5d]
    answer5d = answermapping_5d[most_common_answer_5d]
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {"thinking": thinking5d, "answer": answer5d}
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])
    
    cot_instruction_6 = "Sub-task 6: Analyze the verified final product E to identify all unique carbon environments, considering symmetry, chemical equivalence, and stereochemistry, to accurately predict the number of distinct 13C-NMR signals."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5d", "answer of subtask 5d"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5d, answer5d], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, analyzing final product E carbon environments, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Compare the predicted number of 13C-NMR signals with the provided multiple-choice options (3, 11, 8, 6) and select the correct choice corresponding to the number of signals."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, selecting correct 13C-NMR signal count, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs