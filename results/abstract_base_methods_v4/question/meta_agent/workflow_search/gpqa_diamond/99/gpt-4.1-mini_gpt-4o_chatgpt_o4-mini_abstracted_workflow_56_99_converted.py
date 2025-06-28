async def forward_99(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify the structure and isomeric form of Compound A (C3H6), and determine the product Compound B formed after bromination in carbon tetrachloride, including its structure and stereochemistry."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying Compound A and B, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2 = "Sub-task 2: Determine the structure and properties of Compound C formed when Compound B reacts with alcoholic KOH, specifying the type of reaction and the nature (physical state, flammability) of Compound C, based on Sub-task 1 outputs."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining Compound C, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_reflect_instruction_3 = "Sub-task 3: Identify the possible products (Compound D) formed by passing Compound C through a red-hot iron tube, analyze alternative reaction pathways, and evaluate their structural and spectral characteristics, including 1H NMR features, based on Sub-task 2 outputs. Use a Debate pattern to evaluate alternative products and avoid premature conclusions."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], cot_reflect_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, cot_reflect_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, identifying Compound D and analyzing spectra, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make final decision on Compound D identification and spectral characteristics.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining Compound D and spectra, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Determine the structure and properties of Compound E formed when Compound D reacts with a mixture of two strong acids, specifying the reaction type and expected functional groups, based on Sub-task 3 outputs."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, determining Compound E, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Identify Compound F formed when Compound E reacts with iron scrap and hydrochloric acid, and analyze its common uses, particularly in dye synthesis, debating the possibilities based on Sub-task 4 outputs."
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
            agents.append(f"Debate agent {agent.id}, round {r}, identifying Compound F and uses, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on Compound F identification and uses.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining Compound F and uses, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6 = "Sub-task 6: Determine the structure and properties of Compound G formed when Compound F reacts with nitrous acid, including the type of functional group transformation involved, based on Sub-task 5 outputs."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, determining Compound G, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Identify Compound H formed when Compound G reacts with sodium hydroxide, analyze its chemical behavior, including its reaction with ferric chloride solution and expected color changes, based on Sub-task 6 outputs."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, determining Compound H and analyzing color reactions, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Sub-task 8: Analyze the 1H NMR spectral characteristics of Compound D to verify the claim that it gives two singlets, considering the structural isomer(s) identified in Subtask 3."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking3, answer3], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, analyzing 1H NMR of Compound D, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    cot_instruction_9 = "Sub-task 9: Evaluate the claim that Compound F is used for the synthesis of dyes based on its structure, known chemical properties, and industrial applications, based on Sub-task 5 outputs."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking5, answer5], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, evaluating dye synthesis claim of Compound F, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    cot_instruction_10 = "Sub-task 10: Assess whether Compound H gives a yellow color with ferric chloride solution by analyzing its functional groups and known qualitative tests, based on Sub-task 7 outputs."
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction_10,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking7, answer7], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, assessing color reaction of Compound H, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    
    cot_instruction_11 = "Sub-task 11: Determine if Compound C is a flammable gas by analyzing its molecular structure, physical state, and flammability properties, based on Sub-task 2 outputs."
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_instruction_11,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking11, answer11 = await cot_agent_11([taskInfo, thinking2, answer2], cot_instruction_11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_11.id}, determining flammability of Compound C, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])
    
    cot_sc_instruction_12 = "Sub-task 12: Enumerate all incorrect statements among the given choices by integrating and cross-validating findings from Sub-tasks 8, 9, 10, and 11, using a Self-Consistency Chain-of-Thought approach to handle conflicting conclusions and ensure logical consistency. Explicitly allow multiple incorrect statements to be identified."
    cot_agents_12 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_12 = []
    thinkingmapping_12 = {}
    answermapping_12 = {}
    subtask_desc12 = {
        "subtask_id": "subtask_12",
        "instruction": cot_sc_instruction_12,
        "context": ["user query", "thinking and answer of subtask 8", "thinking and answer of subtask 9", "thinking and answer of subtask 10", "thinking and answer of subtask 11"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking12, answer12 = await cot_agents_12[i]([taskInfo, thinking8, answer8, thinking9, answer9, thinking10, answer10, thinking11, answer11], cot_sc_instruction_12, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_12[i].id}, enumerating incorrect statements, thinking: {thinking12.content}; answer: {answer12.content}")
        possible_answers_12.append(answer12.content)
        thinkingmapping_12[answer12.content] = thinking12
        answermapping_12[answer12.content] = answer12
    most_common_answer_12 = Counter(possible_answers_12).most_common(1)[0][0]
    thinking12 = thinkingmapping_12[most_common_answer_12]
    answer12 = answermapping_12[most_common_answer_12]
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    subtask_desc12['response'] = {"thinking": thinking12, "answer": answer12}
    logs.append(subtask_desc12)
    print("Step 12: ", sub_tasks[-1])
    
    cot_instruction_13 = "Sub-task 13: If the question demands a single incorrect choice, apply predefined criteria (e.g., strongest evidence, most significant error) to select the most appropriate incorrect statement from those identified in Sub-task 12; otherwise, report all incorrect statements."
    cot_agent_13 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc13 = {
        "subtask_id": "subtask_13",
        "instruction": cot_instruction_13,
        "context": ["user query", "thinking and answer of subtask 12"],
        "agent_collaboration": "CoT"
    }
    thinking13, answer13 = await cot_agent_13([taskInfo, thinking12, answer12], cot_instruction_13, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_13.id}, selecting final incorrect statement(s), thinking: {thinking13.content}; answer: {answer13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking13.content}; answer - {answer13.content}")
    subtask_desc13['response'] = {"thinking": thinking13, "answer": answer13}
    logs.append(subtask_desc13)
    print("Step 13: ", sub_tasks[-1])
    
    cot_instruction_14 = "Sub-task 14: Perform a meta-validation step to cross-check that the number and identity of incorrect statements reported align with the conclusions drawn in previous subtasks, flagging any inconsistencies for review before final output."
    cot_agent_14 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc14 = {
        "subtask_id": "subtask_14",
        "instruction": cot_instruction_14,
        "context": ["user query", "thinking and answer of subtask 12", "thinking and answer of subtask 13"],
        "agent_collaboration": "CoT"
    }
    thinking14, answer14 = await cot_agent_14([taskInfo, thinking12, answer12, thinking13, answer13], cot_instruction_14, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_14.id}, performing meta-validation, thinking: {thinking14.content}; answer: {answer14.content}")
    sub_tasks.append(f"Sub-task 14 output: thinking - {thinking14.content}; answer - {answer14.content}")
    subtask_desc14['response'] = {"thinking": thinking14, "answer": answer14}
    logs.append(subtask_desc14)
    print("Step 14: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking14, answer14, sub_tasks, agents)
    return final_answer, logs