async def forward_33(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the general mechanism of the Pinacol rearrangement reaction, detailing the protonation of vicinal diols, formation of carbocation intermediates, and the principles governing group migration including migratory aptitudes (aryl vs. alkyl) and carbocation stability under acidic conditions."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing Pinacol rearrangement mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_reflect_instruction_2a = "Sub-task 2a: For each given vicinal diol compound (A, B, and C), identify all possible protonation sites and corresponding carbocation intermediates, explicitly enumerating each intermediate with structural representations (e.g., SMILES or 2D sketches) to facilitate detailed mechanistic analysis."
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_reflect_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2a, answer2a = await cot_agent_2a([taskInfo, thinking1, answer1], cot_reflect_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, identifying protonation sites and carbocation intermediates, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_reflect_instruction_2b = "Sub-task 2b: Evaluate the relative stabilities of all identified carbocation intermediates for each compound by considering resonance effects, inductive effects, and substituent influences. Perform a reflexive review to confirm the most plausible carbocation intermediates for further analysis."
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2b = self.max_round
    cot_inputs_2b = [taskInfo, thinking1, answer1, thinking2a, answer2a]
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_reflect_instruction_2b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "Reflexion"
    }
    thinking2b, answer2b = await cot_agent_2b(cot_inputs_2b, cot_reflect_instruction_2b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, evaluating carbocation stabilities, thinking: {thinking2b.content}; answer: {answer2b.content}")
    for i in range(N_max_2b):
        feedback, correct = await critic_agent_2b([taskInfo, thinking2b, answer2b], "Please review the carbocation stability evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2b.extend([thinking2b, answer2b, feedback])
        thinking2b, answer2b = await cot_agent_2b(cot_inputs_2b, cot_reflect_instruction_2b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, refining carbocation stability evaluation, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    
    N = self.max_sc
    cot_sc_instruction_4a = "Sub-task 4: For compound A (3-methyl-4-phenylhexane-3,4-diol), enumerate all possible migrating groups from the most stable carbocation intermediate(s), compare their migratory aptitudes (aryl vs. alkyl), and predict all plausible rearranged ketone products with detailed structural validation and IUPAC naming."
    cot_sc_instruction_5b = "Sub-task 5: For compound B (3-(4-hydroxyphenyl)-2-phenylpentane-2,3-diol), enumerate all possible migrating groups from the most stable carbocation intermediate(s), rigorously compare migratory aptitudes including the effect of the p-hydroxy substituent on aryl migration, and predict all plausible rearranged ketone products with structural validation and correct naming."
    cot_sc_instruction_6c = "Sub-task 6: For compound C (1,1,2-tris(4-methoxyphenyl)-2-phenylethane-1,2-diol), enumerate all possible migrating groups from the most stable carbocation intermediate(s), assess migratory aptitudes considering multiple aryl substituents, predict all plausible rearranged ketone products, and validate structures and names thoroughly."
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_5b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_6c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    possible_answers_5b = []
    thinkingmapping_5b = {}
    answermapping_5b = {}
    possible_answers_6c = []
    thinkingmapping_6c = {}
    answermapping_6c = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5b,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6c,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking2b, answer2b], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, enumerating migrations and predicting products for compound A, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
        thinking5b, answer5b = await cot_agents_5b[i]([taskInfo, thinking2b, answer2b], cot_sc_instruction_5b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5b[i].id}, enumerating migrations and predicting products for compound B, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers_5b.append(answer5b.content)
        thinkingmapping_5b[answer5b.content] = thinking5b
        answermapping_5b[answer5b.content] = answer5b
        thinking6c, answer6c = await cot_agents_6c[i]([taskInfo, thinking2b, answer2b], cot_sc_instruction_6c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6c[i].id}, enumerating migrations and predicting products for compound C, thinking: {thinking6c.content}; answer: {answer6c.content}")
        possible_answers_6c.append(answer6c.content)
        thinkingmapping_6c[answer6c.content] = thinking6c
        answermapping_6c[answer6c.content] = answer6c
    most_common_answer_4a = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[most_common_answer_4a]
    answer4a = answermapping_4a[most_common_answer_4a]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    most_common_answer_5b = Counter(possible_answers_5b).most_common(1)[0][0]
    thinking5b = thinkingmapping_5b[most_common_answer_5b]
    answer5b = answermapping_5b[most_common_answer_5b]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    most_common_answer_6c = Counter(possible_answers_6c).most_common(1)[0][0]
    thinking6c = thinkingmapping_6c[most_common_answer_6c]
    answer6c = answermapping_6c[most_common_answer_6c]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6c.content}; answer - {answer6c.content}")
    subtask_desc6['response'] = {"thinking": thinking6c, "answer": answer6c}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    debate_instruction_7 = "Sub-task 7: Conduct a debate-style evaluation comparing all predicted products from compounds A, B, and C, weighing mechanistic plausibility, migratory aptitude rules, and structural correctness to select the most chemically reasonable product for each compound."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking4a, answer4a, thinking5b, answer5b, thinking6c, answer6c], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking4a, answer4a, thinking5b, answer5b, thinking6c, answer6c] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating predicted products, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the most chemically reasonable products for compounds A, B, and C.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding most reasonable products, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Sub-task 8: Cross-validate the selected products against the given multiple-choice options, ensuring exact structural and nomenclature matches, and finalize the correct choice corresponding to the Pinacol rearrangement products of the given compounds."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, cross-validating products with choices, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    cot_reflect_instruction_9 = "Sub-task 9: Perform a reflexive review of the entire prediction workflow, verifying consistency in mechanistic reasoning, structural representations, migratory aptitude considerations, and final product selection to prevent error propagation and confirm the reliability of the final answer."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round
    cot_inputs_9 = [taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b, thinking4a, answer4a, thinking5b, answer5b, thinking6c, answer6c, thinking7, answer7, thinking8, answer8]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction_9,
        "context": ["user query", "all previous thinking and answers"],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, reviewing entire workflow, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max_9):
        feedback, correct = await critic_agent_9([taskInfo, thinking9, answer9], "Please review the entire workflow for consistency and reliability.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_9.extend([thinking9, answer9, feedback])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining workflow review, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
