async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []
    cot_instruction_1 = "Sub-task 1: Identify and extract the reactants and conditions for each Michael addition reaction from the query."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying reactants and conditions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    cot_instruction_2 = "Sub-task 2: Extract the given choices for the products of the reactions from the query."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, extracting product choices, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    cot_instruction_3a = "Sub-task 3a: Determine the dominant tautomer of reactant C under the given reaction conditions."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking1, answer1], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, determining dominant tautomer, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    cot_instruction_3b = "Sub-task 3b: Identify the likely nucleophilic form of each reactant after deprotonation/tautomerization."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking1, answer1, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, identifying nucleophilic form, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    cot_reflect_instruction_3c = "Sub-task 3c: Perform the Michael addition on the identified nucleophilic form."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_3c = self.max_round
    cot_inputs_3c = [taskInfo, thinking3b, answer3b]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, performing Michael addition, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max_3c):
        feedback, correct = await critic_agent_3c([taskInfo, thinking3c, answer3c], 
                                       "please review the Michael addition and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining Michael addition, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    cot_sc_instruction_4 = "Sub-task 4: Classify the extracted product choices from subtask_2 based on their structural compatibility with the expected products determined in subtask_3c."
    N_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2, thinking3c, answer3c], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, classifying product choices, thinking: {thinking4.content}; answer: {answer4.content}")
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
    debate_instruction_5 = "Sub-task 5: Compare the expected products with the classified choices to identify the correct choice that matches the expected products."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
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
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing products, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 5: Make final decision on correct product choice.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct product choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    cot_instruction_6 = "Sub-task 6: Select and return the alphabet choice (A, B, C, or D) that corresponds to the correct product choice identified in subtask_5."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting alphabet choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
