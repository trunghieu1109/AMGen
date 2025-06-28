async def forward_78(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the molecular formula of Compound X (C11H12O) to determine degrees of unsaturation and infer possible functional groups consistent with this formula." 
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular formula, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Interpret the reaction conditions (reaction with 1,4-diazabicyclo[2.2.2]octane in a nonpolar solvent at 403 K for 1 day) to hypothesize the likely chemical transformation undergone by Compound X, considering the reagents known reactivity, based on Sub-task 1 output." 
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, interpreting reaction conditions, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_sc_instruction_3 = "Sub-task 3: Perform a detailed analysis of the 1H NMR spectral data: assign chemical shifts, integration, and multiplicity to specific proton environments, explicitly checking for signals characteristic of epoxide protons (2.5–3.5 ppm) and aromatic substitution patterns, and generate multiple hypotheses about proton environments, based on Sub-task 2 output." 
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
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing 1H NMR data, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction_4 = "Sub-task 4: Perform a detailed analysis of the 13C NMR spectral data: assign chemical shifts to carbon environments, explicitly identify signals indicative of carbonyl carbons (~190–210 ppm) and epoxide carbons (50–60 ppm), and generate multiple hypotheses about carbon environments, based on Sub-task 2 output." 
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyzing 13C NMR data, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_reflect_instruction_5 = "Sub-task 5: Reflexively evaluate and cross-validate the 1H and 13C NMR analyses from Sub-tasks 3 and 4 to confirm or refute the presence of key functional groups (epoxide vs. ketone), reconcile inconsistencies, and refine structural hypotheses accordingly." 
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, evaluating NMR analyses, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Please review the NMR functional group assignments and structural hypotheses, provide limitations or inconsistencies.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining NMR evaluation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction_6a = "Sub-task 6a: Correlate the refined NMR-based structural hypotheses with the molecular formula and reaction conditions to deduce plausible product structures, focusing on functional group compatibility and substitution patterns, based on Sub-task 5 output." 
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking5, answer5], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, deducing plausible product structures, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {
        "thinking": thinking6a,
        "answer": answer6a
    }
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    debate_instruction_6b = "Sub-task 6b: Systematically compare each multiple-choice option (2-styrylepoxide, 2-(4-methylstyryl)oxirane, 2-(1-phenylprop-1-en-2-yl)oxirane, 2-methyl-3-styryloxirane) against the deduced structural features and spectral data, explicitly checking for consistency with key NMR signals and functional groups." 
    debate_agents_6b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6b = self.max_round
    all_thinking6b = [[] for _ in range(N_max_6b)]
    all_answer6b = [[] for _ in range(N_max_6b)]
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": debate_instruction_6b,
        "context": ["user query", "thinking of subtask 6a", "answer of subtask 6a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6b):
        for i, agent in enumerate(debate_agents_6b):
            if r == 0:
                thinking6b, answer6b = await agent([taskInfo, thinking6a, answer6a], debate_instruction_6b, r, is_sub_task=True)
            else:
                input_infos_6b = [taskInfo, thinking6a, answer6a] + all_thinking6b[r-1] + all_answer6b[r-1]
                thinking6b, answer6b = await agent(input_infos_6b, debate_instruction_6b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing candidate structures, thinking: {thinking6b.content}; answer: {answer6b.content}")
            all_thinking6b[r].append(thinking6b)
            all_answer6b[r].append(answer6b)
    final_decision_agent_6c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6c, answer6c = await final_decision_agent_6c([taskInfo] + all_thinking6b[-1] + all_answer6b[-1], "Sub-task 6c: Identify and flag inconsistencies or mismatches between spectral data and candidate structures, and select the multiple-choice option that best fits all evidence.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting best matching multiple-choice option, thinking: {thinking6c.content}; answer: {answer6c.content}")
    sub_tasks.append(f"Sub-task 6c output: thinking - {thinking6c.content}; answer - {answer6c.content}")
    subtask_desc6c = {
        "thinking": thinking6c,
        "answer": answer6c
    }
    subtask_desc6c.update({
        "subtask_id": "subtask_6c",
        "instruction": "Sub-task 6c: Identify and flag inconsistencies or mismatches between spectral data and candidate structures, and select the best multiple-choice option.",
        "context": ["user query", "thinking of subtask 6b", "answer of subtask 6b"],
        "agent_collaboration": "Final Decision"
    })
    logs.append(subtask_desc6c)
    print("Step 6c: ", sub_tasks[-1])
    cot_instruction_7 = "Sub-task 7: Validate the final selected answer for consistency with all spectral data, molecular formula, reaction conditions, and instructions, ensuring adherence to the required output format (return only the letter choice: A, B, C, or D)." 
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6c", "answer of subtask 6c"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6c, answer6c], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, validating final answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs