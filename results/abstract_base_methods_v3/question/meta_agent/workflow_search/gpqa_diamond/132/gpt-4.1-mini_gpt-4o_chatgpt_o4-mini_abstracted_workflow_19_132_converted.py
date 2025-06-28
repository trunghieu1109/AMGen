async def forward_132(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_1 = "Sub-task 1: Calculate the number of moles of KH2PO4 and Na2HPO4·2H2O in the solution using their given masses and molecular weights."
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, calculating moles of KH2PO4 and Na2HPO4·2H2O, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    most_common_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[most_common_answer_1]
    answer1 = answermapping_1[most_common_answer_1]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    cot_instruction_2 = "Sub-task 2: Convert the solution volume from cm³ to liters to enable concentration calculations."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, converting volume from cm3 to liters, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    cot_sc_instruction_3 = "Sub-task 3: Calculate the molar concentrations of KH2PO4 and Na2HPO4·2H2O in the solution by dividing their moles by the solution volume in liters."
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculating molar concentrations, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Identify the phosphate species present in solution contributed by KH2PO4 (H2PO4⁻) and Na2HPO4·2H2O (HPO4²⁻), and summarize their relevant dissociation equilibria using the given Ka1, Ka2, and Ka3 values for H3PO4 at 25 °C."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, identifying phosphate species and equilibria, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Set up the equilibrium expressions for the dissociation of phosphate species in solution, incorporating the initial concentrations of H2PO4⁻ and HPO4²⁻ and the given Ka values."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, setting up equilibrium expressions, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on equilibrium expressions.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding equilibrium expressions, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    cot_sc_instruction_6a = "Sub-task 6a: Calculate the solution pH by applying the Henderson–Hasselbalch equation using Ka2 and the ratio of initial concentrations of HPO4²⁻ and H2PO4⁻."
    cot_agents_6a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6a = []
    thinkingmapping_6a = {}
    answermapping_6a = {}
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_sc_instruction_6a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6a, answer6a = await cot_agents_6a[i]([taskInfo, thinking3, answer3, thinking5, answer5], cot_sc_instruction_6a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6a[i].id}, calculating pH using Henderson–Hasselbalch equation, thinking: {thinking6a.content}; answer: {answer6a.content}")
        possible_answers_6a.append(answer6a.content)
        thinkingmapping_6a[answer6a.content] = thinking6a
        answermapping_6a[answer6a.content] = answer6a
    most_common_answer_6a = Counter(possible_answers_6a).most_common(1)[0][0]
    thinking6a = thinkingmapping_6a[most_common_answer_6a]
    answer6a = answermapping_6a[most_common_answer_6a]
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Subtask 6a answer: ", sub_tasks[-1])
    
    cot_instruction_6b = "Sub-task 6b: Calculate the hydrogen ion concentration [H⁺] explicitly from the pH obtained in subtask 6a."
    cot_agent_6b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_instruction_6b,
        "context": ["user query", "thinking of subtask 6a", "answer of subtask 6a"],
        "agent_collaboration": "CoT"
    }
    thinking6b, answer6b = await cot_agent_6b([taskInfo, thinking6a, answer6a], cot_instruction_6b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6b.id}, calculating [H+] from pH, thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Subtask 6b answer: ", sub_tasks[-1])
    
    cot_sc_instruction_6c = "Sub-task 6c: Calculate the orthophosphate ion concentration [PO4³⁻] using the equilibrium expression for the third dissociation (Ka3), the concentration of HPO4²⁻, and the hydrogen ion concentration [H⁺] from subtask 6b."
    cot_agents_6c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6c = []
    thinkingmapping_6c = {}
    answermapping_6c = {}
    subtask_desc6c = {
        "subtask_id": "subtask_6c",
        "instruction": cot_sc_instruction_6c,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 6b", "answer of subtask 6b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6c, answer6c = await cot_agents_6c[i]([taskInfo, thinking3, answer3, thinking6b, answer6b], cot_sc_instruction_6c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6c[i].id}, calculating orthophosphate ion concentration, thinking: {thinking6c.content}; answer: {answer6c.content}")
        possible_answers_6c.append(answer6c.content)
        thinkingmapping_6c[answer6c.content] = thinking6c
        answermapping_6c[answer6c.content] = answer6c
    most_common_answer_6c = Counter(possible_answers_6c).most_common(1)[0][0]
    thinking6c = thinkingmapping_6c[most_common_answer_6c]
    answer6c = answermapping_6c[most_common_answer_6c]
    sub_tasks.append(f"Sub-task 6c output: thinking - {thinking6c.content}; answer - {answer6c.content}")
    subtask_desc6c['response'] = {"thinking": thinking6c, "answer": answer6c}
    logs.append(subtask_desc6c)
    print("Subtask 6c answer: ", sub_tasks[-1])
    
    debate_instruction_6d = "Sub-task 6d: Validate the calculated orthophosphate ion concentration [PO4³⁻] by having multiple agents independently compute and debate the result to ensure chemical accuracy and consistency."
    debate_agents_6d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6d = self.max_round
    all_thinking6d = [[] for _ in range(N_max_6d)]
    all_answer6d = [[] for _ in range(N_max_6d)]
    subtask_desc6d = {
        "subtask_id": "subtask_6d",
        "instruction": debate_instruction_6d,
        "context": ["user query", "thinking of subtask 6c", "answer of subtask 6c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6d):
        for i, agent in enumerate(debate_agents_6d):
            if r == 0:
                thinking6d, answer6d = await agent([taskInfo, thinking6c, answer6c], debate_instruction_6d, r, is_sub_task=True)
            else:
                input_infos_6d = [taskInfo, thinking6c, answer6c] + all_thinking6d[r-1] + all_answer6d[r-1]
                thinking6d, answer6d = await agent(input_infos_6d, debate_instruction_6d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating orthophosphate concentration, thinking: {thinking6d.content}; answer: {answer6d.content}")
            all_thinking6d[r].append(thinking6d)
            all_answer6d[r].append(answer6d)
    final_decision_agent_6d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6d, answer6d = await final_decision_agent_6d([taskInfo] + all_thinking6d[-1] + all_answer6d[-1], "Sub-task 6d: Make final decision on validated orthophosphate ion concentration.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding validated orthophosphate concentration, thinking: {thinking6d.content}; answer: {answer6d.content}")
    sub_tasks.append(f"Sub-task 6d output: thinking - {thinking6d.content}; answer - {answer6d.content}")
    subtask_desc6d['response'] = {"thinking": thinking6d, "answer": answer6d}
    logs.append(subtask_desc6d)
    print("Subtask 6d answer: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Compare the calculated orthophosphate ion concentration [PO4³⁻] from subtask 6d with the provided multiple-choice options and select the correct answer."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6d", "answer of subtask 6d"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6d, answer6d], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, selecting correct multiple-choice answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Subtask 7 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
