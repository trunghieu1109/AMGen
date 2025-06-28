async def forward_156(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    global_instructions = (
        "Global Instructions: For all subtasks, strictly produce output as a single letter (A, B, C, or D) corresponding to the multiple-choice options provided. "
        "Emphasize quick and accurate detection of the retrovirus as the key requirement. "
        "All reasoning and outputs must align with these constraints."
    )

    # Stage 1: Incorporate global instructions and analyze retrovirus outbreak context

    # Sub-task 1: Explicitly state and incorporate global instructions
    cot_instruction_1 = (
        "Sub-task 1: Explicitly state and incorporate the global instructions, including the requirement to produce a single-letter multiple-choice answer (A, B, C, or D) "
        "and the emphasis on quick and accurate detection, to guide all downstream subtasks. "
        "Output must be strictly the letter A, B, C, or D."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query", "global instructions"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo, global_instructions], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, incorporating global instructions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze retrovirus outbreak context to identify and characterize virus
    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the retrovirus outbreak context to identify and characterize the virus, including obtaining viral genetic material (RNA) "
        "and relevant immunological markers from patient samples to enable evaluation of diagnostic approaches. "
        "Remember to strictly produce output as a single letter (A, B, C, or D) and emphasize quick detection."
    )
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "global instructions"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1, global_instructions], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing retrovirus outbreak, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content.strip().upper())
        thinkingmapping_2[answer2.content.strip().upper()] = thinking2
        answermapping_2[answer2.content.strip().upper()] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Evaluate PCR-based and ELISA-based diagnostic methods

    # Sub-task 3: Evaluate PCR-based diagnostic methods
    cot_sc_instruction_3 = (
        "Sub-task 3: Evaluate PCR-based diagnostic methods by: (a) converting viral RNA to cDNA via reverse transcription, "
        "(b) sequencing cDNA to identify viral genome sequences, (c) designing primers and probes for real-time PCR assays, "
        "and (d) assessing their suitability for quick and accurate detection of the retrovirus. "
        "Output must be a single letter (A, B, C, or D) strictly."
    )
    N3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "global instructions"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N3):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2, global_instructions], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, evaluating PCR-based methods, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content.strip().upper())
        thinkingmapping_3[answer3.content.strip().upper()] = thinking3
        answermapping_3[answer3.content.strip().upper()] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate ELISA-based diagnostic methods
    cot_sc_instruction_4 = (
        "Sub-task 4: Evaluate ELISA-based diagnostic methods by: (a) identifying IgG antibodies specific to the retrovirus in patient samples, "
        "(b) designing ELISA kits targeting these antibodies, and (c) assessing their suitability for quick and accurate detection of the retrovirus. "
        "Output must be a single letter (A, B, C, or D) strictly."
    )
    N4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "global instructions"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2, global_instructions], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, evaluating ELISA-based methods, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content.strip().upper())
        thinkingmapping_4[answer4.content.strip().upper()] = thinking4
        answermapping_4[answer4.content.strip().upper()] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Compare methods and select final answer

    # Sub-task 5: Compare and reflect on PCR vs ELISA methods
    debate_instruction_5 = (
        "Sub-task 5: Compare and reflect on the advantages and limitations of PCR-based versus ELISA-based diagnostic methods "
        "in the context of quick and accurate detection, considering the retrovirus characteristics and outbreak requirements. "
        "Output must be a single letter (A, B, C, or D) strictly."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4", "global instructions"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4, global_instructions], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4, global_instructions] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing PCR vs ELISA, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1] + [global_instructions], "Sub-task 5: Make final decision on the most appropriate diagnostic approach considering quick and accurate detection, output only the letter A, B, C, or D.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final diagnostic approach selection, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Enforce output format compliance and produce final answer
    cot_instruction_6 = (
        "Sub-task 6: Based on the comparative evaluation, select the most appropriate diagnostic approach and corresponding multiple-choice option (A, B, C, or D). "
        "Strictly output only the single letter choice without any additional text."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5", "global instructions"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5, global_instructions], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, enforcing output format and final answer selection, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
