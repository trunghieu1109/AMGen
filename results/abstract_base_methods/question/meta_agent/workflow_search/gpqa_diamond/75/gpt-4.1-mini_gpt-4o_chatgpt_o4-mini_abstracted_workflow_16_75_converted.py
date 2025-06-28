async def forward_75(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Understand the experimental setup: identify the source of iPSCs (somatic cells from various tissues), labeling strategy (mRaspberry under lineage-specific promoters), injection stage (32-cell blastocyst), observation timeline (48 hours with sampling every 12 hours), and staining method (TUNEL-FITC for apoptosis)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding experimental setup, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Define and characterize the biological markers: specify the red fluorescent signal as mRaspberry expressed in iPSCs under lineage-specific promoters (expected cytoplasmic/organellar localization) and the green fluorescent signal as TUNEL-FITC staining indicating apoptotic nuclear DNA fragmentation (expected nuclear localization), based on Sub-task 1 outputs."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, defining biological markers, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_instruction_3 = "Sub-task 3: Analyze the expected spatial and subcellular localization patterns of the red mRaspberry signal in the embryo, considering lineage specificity and typical cytoplasmic or organelle-associated distribution in iPSCs and their derivatives, based on Sub-task 2 outputs."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing red signal localization, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Analyze the expected temporal appearance and subcellular localization of the green TUNEL-FITC apoptotic signal in the embryo over the 48-hour period, including the known developmental timing of apoptosis onset in mouse blastocyst chimeras (likely delayed relative to injection), based on Sub-task 2 outputs."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing green signal localization and timing, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_reflect_instruction_5 = "Sub-task 5: Assess the spatial relationship between red (mRaspberry) and green (TUNEL-FITC) signals in confocal microscopy images at each 12-hour time point, explicitly considering the distinct subcellular localizations (cytoplasmic/organellar red vs. nuclear green) and the resolution limits of confocal microscopy to avoid false assumptions of co-localization, based on Sub-tasks 3 and 4 outputs."
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
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, assessing spatial relationship, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please review the spatial relationship assessment and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining spatial relationship, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6a = "Sub-task 6a: Critically evaluate the biological plausibility of co-localization of red and green signals as the first observable event, given the distinct subcellular localizations and known timing of apoptosis in the embryo, based on Sub-task 5 output."
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking5, answer5], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, evaluating biological plausibility of co-localization, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {
        "thinking": thinking6a,
        "answer": answer6a
    }
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    
    cot_instruction_6b = "Sub-task 6b: Systematically evaluate each multiple-choice option (A-D) against experimental facts, biological principles, and imaging constraints to identify the most likely first observable event: (A) green signal colocalizes with red signal, (B) no green signal present, (C) cell line-specific red signals label different organelles, or (D) cytoplasmic localization of the red signal, based on Sub-task 6a output."
    N = self.max_sc
    cot_agents_6b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6b = []
    thinkingmapping_6b = {}
    answermapping_6b = {}
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_instruction_6b,
        "context": ["user query", "thinking of subtask 6a", "answer of subtask 6a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6b, answer6b = await cot_agents_6b[i]([taskInfo, thinking6a, answer6a], cot_instruction_6b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6b[i].id}, evaluating multiple-choice options, thinking: {thinking6b.content}; answer: {answer6b.content}")
        possible_answers_6b.append(answer6b.content)
        thinkingmapping_6b[answer6b.content] = thinking6b
        answermapping_6b[answer6b.content] = answer6b
    answer6b_content = Counter(possible_answers_6b).most_common(1)[0][0]
    thinking6b = thinkingmapping_6b[answer6b_content]
    answer6b = answermapping_6b[answer6b_content]
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {
        "thinking": thinking6b,
        "answer": answer6b
    }
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    
    debate_instruction_6c = "Sub-task 6c: Perform a reflective debate or self-consistency check by comparing independent reasonings on the first observable event to reduce anchoring bias and confirm the most biologically accurate interpretation before finalizing the answer, based on Sub-task 6b output."
    debate_agents_6c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6c = self.max_round
    all_thinking6c = [[] for _ in range(N_max_6c)]
    all_answer6c = [[] for _ in range(N_max_6c)]
    subtask_desc6c = {
        "subtask_id": "subtask_6c",
        "instruction": debate_instruction_6c,
        "context": ["user query", "thinking of subtask 6b", "answer of subtask 6b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6c):
        for i, agent in enumerate(debate_agents_6c):
            if r == 0:
                thinking6c, answer6c = await agent([taskInfo, thinking6b, answer6b], debate_instruction_6c, r, is_sub_task=True)
            else:
                input_infos_6c = [taskInfo, thinking6b, answer6b] + all_thinking6c[r-1] + all_answer6c[r-1]
                thinking6c, answer6c = await agent(input_infos_6c, debate_instruction_6c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, reflective debate on first observable event, thinking: {thinking6c.content}; answer: {answer6c.content}")
            all_thinking6c[r].append(thinking6c)
            all_answer6c[r].append(answer6c)
    final_decision_agent_6c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6c, answer6c = await final_decision_agent_6c([taskInfo] + all_thinking6c[-1] + all_answer6c[-1], "Sub-task 6c: Make final decision on the first observable event in the embryo after injection and staining.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on first observable event, thinking: {thinking6c.content}; answer: {answer6c.content}")
    sub_tasks.append(f"Sub-task 6c output: thinking - {thinking6c.content}; answer - {answer6c.content}")
    subtask_desc6c['response'] = {
        "thinking": thinking6c,
        "answer": answer6c
    }
    logs.append(subtask_desc6c)
    print("Step 6c: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6c, answer6c, sub_tasks, agents)
    return final_answer, logs