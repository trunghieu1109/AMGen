async def forward_39(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Define and interpret the phrase 'my compounds are on top of each other' within the context of synthetic organic chemistry laboratory language, focusing on common scenarios and jargon related to compound analysis and separation."
    cot_instruction_1b = "Sub-task 1b: Identify the laboratory technique most likely implied by the phrase, emphasizing common synthetic organic chemistry methods such as Thin Layer Chromatography (TLC), and describe typical failure modes like overlapping spots."
    cot_instruction_1c = "Sub-task 1c: Explain the chemical basis for overlapping spots in the identified technique, particularly how similar compound properties (e.g., polarity) affect separation and cause compounds to appear 'on top of each other.'"
    
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, interpret phrase meaning, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, identify technique and failure modes, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])
    
    N_sc_1c = self.max_sc
    cot_agents_1c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1c)]
    possible_answers_1c = []
    thinkingmapping_1c = {}
    answermapping_1c = {}
    subtask_desc1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1c):
        thinking1c, answer1c = await cot_agents_1c[i]([taskInfo, thinking1b, answer1b], cot_instruction_1c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1c[i].id}, explain chemical basis for overlapping spots, thinking: {thinking1c.content}; answer: {answer1c.content}")
        possible_answers_1c.append(answer1c.content)
        thinkingmapping_1c[answer1c.content] = thinking1c
        answermapping_1c[answer1c.content] = answer1c
    answer1c_content = Counter(possible_answers_1c).most_common(1)[0][0]
    thinking1c = thinkingmapping_1c[answer1c_content]
    answer1c = answermapping_1c[answer1c_content]
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc1c['response'] = {"thinking": thinking1c, "answer": answer1c}
    logs.append(subtask_desc1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Analyze each multiple-choice option by defining the chemical concept it represents (non-covalent interactions, polarity similarity, optical rotation similarity, boiling point similarity) and assess its relevance to the overlapping compounds scenario in the context of the identified technique."
    N_sc_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1c, answer1c], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyze multiple-choice options, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Conduct a critical comparison and debate of the plausible explanations for the phrase, using domain knowledge to select the most contextually appropriate interpretation and discard less likely options."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating plausible explanations, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 6: Perform a reflexive review of the reasoning process to identify and correct any misinterpretations or errors before finalizing the answer choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, reflexive review and correction, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": "Sub-task 6: Perform a reflexive review of the reasoning process to identify and correct any misinterpretations or errors before finalizing the answer choice.",
        "context": ["user query", "thinking of subtask_5", "answer of subtask_5"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Select the correct multiple-choice answer (A, B, C, or D) based on the validated reasoning and provide a concise justification aligned with synthetic organic chemistry laboratory practices."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask_6", "answer of subtask_6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, select final answer and justification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs