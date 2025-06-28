async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Define variables representing each digit in the 2x3 grid (a, b, c in the first row; d, e, f in the second row) and express the two horizontal numbers formed by reading each row left to right as algebraic expressions."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, define digit variables and horizontal numbers, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Formulate the equation representing the sum of the two horizontal numbers (from subtask_1) equaling 999."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, formulate horizontal sum equation, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_instruction_3 = "Sub-task 3: Express the three vertical numbers formed by reading each column top to bottom as algebraic expressions using the digit variables."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, express vertical numbers, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4 = "Sub-task 4: Formulate the equation representing the sum of the three vertical numbers (from subtask_3) equaling 99."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, formulate vertical sum equation, thinking: {thinking4.content}; answer: {answer4.content}")
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
    
    debate_instruction_5 = "Sub-task 5: Combine the horizontal and vertical sum equations (from subtasks 2 and 4) into a system of equations and derive all algebraic constraints on the digits without making premature assumptions; explicitly include digit range constraints (each digit 0-9). Verify these constraints against the example grid to ensure consistency."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking2, answer2, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking2, answer2, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyze system of equations and constraints, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the system of equations and digit constraints.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing system analysis, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6a = "Sub-task 6a: Analyze the combined system to identify all valid pairs (a, d) of digits that satisfy the horizontal sum equation and digit constraints, enumerating all possibilities without restricting to a=9 and d=9 only."
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking5, answer5], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, enumerate valid (a, d) pairs, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {
        "thinking": thinking6a,
        "answer": answer6a
    }
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    
    cot_instruction_6b = "Sub-task 6b: For each valid (a, d) pair from subtask_6a, analyze the vertical sum equation to derive constraints on digits b, c, e, f, ensuring all digit values are between 0 and 9 and that the equation 10b + 10c + e + f = 9 is correctly interpreted with valid digit assignments."
    cot_agent_6b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_instruction_6b,
        "context": ["user query", "thinking of subtask 6a", "answer of subtask 6a"],
        "agent_collaboration": "CoT"
    }
    thinking6b, answer6b = await cot_agent_6b([taskInfo, thinking6a, answer6a], cot_instruction_6b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6b.id}, analyze vertical sum constraints for b, c, e, f, thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {
        "thinking": thinking6b,
        "answer": answer6b
    }
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    
    cot_instruction_6c = "Sub-task 6c: Enumerate all valid quadruples (b, c, e, f) for each (a, d) pair that satisfy the vertical sum constraints and digit range restrictions, using a structured approach: first identify valid (b, c) pairs such that 10b + 10c ≤ 9, then for each, find (e, f) pairs such that e + f = 9 - 10b - 10c."
    cot_agent_6c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6c = {
        "subtask_id": "subtask_6c",
        "instruction": cot_instruction_6c,
        "context": ["user query", "thinking of subtask 6b", "answer of subtask 6b"],
        "agent_collaboration": "CoT"
    }
    thinking6c, answer6c = await cot_agent_6c([taskInfo, thinking6b, answer6b], cot_instruction_6c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6c.id}, enumerate valid (b, c, e, f) quadruples, thinking: {thinking6c.content}; answer: {answer6c.content}")
    sub_tasks.append(f"Sub-task 6c output: thinking - {thinking6c.content}; answer - {answer6c.content}")
    subtask_desc6c['response'] = {
        "thinking": thinking6c,
        "answer": answer6c
    }
    logs.append(subtask_desc6c)
    print("Step 6c: ", sub_tasks[-1])
    
    cot_reflect_instruction_6d = "Sub-task 6d: Cross-verify all enumerated digit assignments (a, b, c, d, e, f) against both the horizontal and vertical sum equations and the example grid to ensure correctness and consistency; if inconsistencies are found, trigger reflexion to reassess constraints and enumeration."
    cot_agent_6d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6d = self.max_round
    cot_inputs_6d = [taskInfo, thinking6c, answer6c]
    subtask_desc6d = {
        "subtask_id": "subtask_6d",
        "instruction": cot_reflect_instruction_6d,
        "context": ["user query", "thinking of subtask 6c", "answer of subtask 6c"],
        "agent_collaboration": "Reflexion"
    }
    thinking6d, answer6d = await cot_agent_6d(cot_inputs_6d, cot_reflect_instruction_6d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6d.id}, cross-verify enumerations, thinking: {thinking6d.content}; answer: {answer6d.content}")
    for i in range(N_max_6d):
        feedback, correct = await critic_agent_6d([taskInfo, thinking6d, answer6d], "Please review the enumerated digit assignments for correctness and consistency.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6d.extend([thinking6d, answer6d, feedback])
        thinking6d, answer6d = await cot_agent_6d(cot_inputs_6d, cot_reflect_instruction_6d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6d.id}, refining verification, thinking: {thinking6d.content}; answer: {answer6d.content}")
    sub_tasks.append(f"Sub-task 6d output: thinking - {thinking6d.content}; answer - {answer6d.content}")
    subtask_desc6d['response'] = {
        "thinking": thinking6d,
        "answer": answer6d
    }
    logs.append(subtask_desc6d)
    print("Step 6d: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Aggregate all valid digit assignments from subtask_6d that satisfy both sum conditions and digit constraints."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6d", "answer of subtask 6d"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6d, answer6d], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, aggregate valid digit assignments, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_sc_instruction_8 = "Sub-task 8: Count the total number of valid digit assignments found in subtask_7 and return this integer as the final answer."
    cot_agents_8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_8 = []
    thinkingmapping_8 = {}
    answermapping_8 = {}
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_sc_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking8, answer8 = await cot_agents_8[i]([taskInfo, thinking7, answer7], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_8[i].id}, count total valid assignments, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers_8.append(answer8.content)
        thinkingmapping_8[answer8.content] = thinking8
        answermapping_8[answer8.content] = answer8
    answer8_content = Counter(possible_answers_8).most_common(1)[0][0]
    thinking8 = thinkingmapping_8[answer8_content]
    answer8 = answermapping_8[answer8_content]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {
        "thinking": thinking8,
        "answer": answer8
    }
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
