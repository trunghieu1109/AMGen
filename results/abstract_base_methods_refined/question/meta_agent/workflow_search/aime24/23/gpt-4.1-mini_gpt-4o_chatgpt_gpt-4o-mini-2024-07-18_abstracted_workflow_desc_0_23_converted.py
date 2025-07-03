async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Define variables representing each digit in the 2x3 grid as a, b, c for the top row and d, e, f for the bottom row. Express the two horizontal numbers formed by reading each row left to right as algebraic expressions using these variables."
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
    cot_sc_instruction_2 = "Sub-task 2: Using the digit variables and horizontal numbers from Sub-task 1, formulate the equation representing the sum of the two horizontal numbers (top row number + bottom row number) equal to 999, using the digit variables."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, formulate horizontal sum equation equals 999, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_instruction_3 = "Sub-task 3: Using the digit variables defined in Sub-task 1, express the three vertical numbers formed by reading each column top to bottom as algebraic expressions."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, express vertical numbers as algebraic expressions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction_4 = "Sub-task 4: Using the vertical numbers from Sub-task 3, formulate the equation representing the sum of the three vertical numbers equal to 99, using the digit variables."
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
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, formulate vertical sum equation equals 99, thinking: {thinking4.content}; answer: {answer4.content}")
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
    debate_instruction_5 = "Sub-task 5: Combine the two equations from Sub-task 2 and Sub-Task 4 to form a system of equations representing the constraints on the digit variables."
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
            agents.append(f"Debate agent {agent.id}, round {r}, combining equations into system, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the system of equations representing the digit constraints.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing system of equations, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_sc_instruction_6a = "Sub-task 6a: Simplify the system of equations from Sub-task 5 by grouping like variables and derive explicit linear relations among digit pairs (e.g., a + d = 9), ensuring to incorporate the digit domain constraints (each digit between 0 and 9, with leading digits a and d between 1 and 9)."
    cot_agents_6a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6a = []
    thinkingmapping_6a = {}
    answermapping_6a = {}
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_sc_instruction_6a,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6a, answer6a = await cot_agents_6a[i]([taskInfo, thinking5, answer5], cot_sc_instruction_6a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6a[i].id}, simplify system and derive linear relations, thinking: {thinking6a.content}; answer: {answer6a.content}")
        possible_answers_6a.append(answer6a.content)
        thinkingmapping_6a[answer6a.content] = thinking6a
        answermapping_6a[answer6a.content] = answer6a
    answer6a_content = Counter(possible_answers_6a).most_common(1)[0][0]
    thinking6a = thinkingmapping_6a[answer6a_content]
    answer6a = answermapping_6a[answer6a_content]
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {
        "thinking": thinking6a,
        "answer": answer6a
    }
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    cot_reflect_instruction_6b = "Sub-task 6b: Analyze the constraints on each digit variable individually, especially clarifying the range and non-zero conditions for leading digits (a and d must be between 1 and 9), and verify consistency of the derived relations from Sub-task 6a."
    cot_agent_6b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6b = self.max_round
    cot_inputs_6b = [taskInfo, thinking6a, answer6a]
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_reflect_instruction_6b,
        "context": ["user query", "thinking of subtask 6a", "answer of subtask 6a"],
        "agent_collaboration": "Reflexion"
    }
    thinking6b, answer6b = await cot_agent_6b(cot_inputs_6b, cot_reflect_instruction_6b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6b.id}, analyze digit constraints and verify consistency, thinking: {thinking6b.content}; answer: {answer6b.content}")
    for i in range(N_max_6b):
        feedback, correct = await critic_agent_6b([taskInfo, thinking6b, answer6b], "Please review the digit constraints and verify their consistency and validity.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6b.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6b.extend([thinking6b, answer6b, feedback])
        thinking6b, answer6b = await cot_agent_6b(cot_inputs_6b, cot_reflect_instruction_6b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6b.id}, refining digit constraints, thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {
        "thinking": thinking6b,
        "answer": answer6b
    }
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    cot_sc_instruction_6c = "Sub-task 6c: Cross-validate the derived constraints and relations from Sub-task 6b by independent reasoning or alternative approaches using Self-Consistency Chain-of-Thought to ensure no contradictions or overlooked conditions remain."
    cot_agents_6c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6c = []
    thinkingmapping_6c = {}
    answermapping_6c = {}
    subtask_desc6c = {
        "subtask_id": "subtask_6c",
        "instruction": cot_sc_instruction_6c,
        "context": ["user query", "thinking of subtask 6b", "answer of subtask 6b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6c, answer6c = await cot_agents_6c[i]([taskInfo, thinking6b, answer6b], cot_sc_instruction_6c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6c[i].id}, cross-validate digit constraints, thinking: {thinking6c.content}; answer: {answer6c.content}")
        possible_answers_6c.append(answer6c.content)
        thinkingmapping_6c[answer6c.content] = thinking6c
        answermapping_6c[answer6c.content] = answer6c
    answer6c_content = Counter(possible_answers_6c).most_common(1)[0][0]
    thinking6c = thinkingmapping_6c[answer6c_content]
    answer6c = answermapping_6c[answer6c_content]
    sub_tasks.append(f"Sub-task 6c output: thinking - {thinking6c.content}; answer - {answer6c.content}")
    subtask_desc6c['response'] = {
        "thinking": thinking6c,
        "answer": answer6c
    }
    logs.append(subtask_desc6c)
    print("Step 6c: ", sub_tasks[-1])
    cot_instruction_7 = "Sub-task 7: Using the fully simplified and verified constraints from Sub-task 6c, systematically enumerate all possible digit assignments to the grid cells that satisfy the system of equations and digit constraints, employing combinatorial counting methods where applicable to avoid exhaustive listing."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6c", "answer of subtask 6c"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6c, answer6c], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, enumerate valid digit assignments using combinatorial methods, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_instruction_8 = "Sub-task 8: Count the total number of valid digit assignments found in Sub-task 7 that satisfy both the horizontal and vertical sum conditions, ensuring the count is accurate and justified by the enumeration method used."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, count total valid assignments, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {
        "thinking": thinking8,
        "answer": answer8
    }
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs