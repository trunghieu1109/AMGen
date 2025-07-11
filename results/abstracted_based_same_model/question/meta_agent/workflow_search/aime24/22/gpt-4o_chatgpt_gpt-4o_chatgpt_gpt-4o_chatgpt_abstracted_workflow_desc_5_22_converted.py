async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1A = "Sub-task 1A: Enumerate possible pairs of (k, n) where k is the frequency of the mode 9, and n is the total number of elements in the list, ensuring the sum of the list is 30."
    cot_agent_1A = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1A = {
        "subtask_id": "subtask_1A",
        "instruction": cot_instruction_1A,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1A, answer1A = await cot_agent_1A([taskInfo], cot_instruction_1A, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1A.id}, enumerating (k, n) pairs, thinking: {thinking1A.content}; answer: {answer1A.content}")
    sub_tasks.append(f"Sub-task 1A output: thinking - {thinking1A.content}; answer - {answer1A.content}")
    subtask_desc1A['response'] = {
        "thinking": thinking1A,
        "answer": answer1A
    }
    logs.append(subtask_desc1A)
    cot_sc_instruction_1B = "Sub-task 1B: For each (k, n) pair from subtask 1A, generate all possible partitions of the remaining sum (30 - 9k) into n-k elements, ensuring the mode is unique and the median is a positive integer not in the list."
    N = self.max_sc
    cot_agents_1B = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1B = []
    thinkingmapping_1B = {}
    answermapping_1B = {}
    subtask_desc1B = {
        "subtask_id": "subtask_1B",
        "instruction": cot_sc_instruction_1B,
        "context": ["user query", "thinking of subtask 1A", "answer of subtask 1A"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1B, answer1B = await cot_agents_1B[i]([taskInfo, thinking1A, answer1A], cot_sc_instruction_1B, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1B[i].id}, generating partitions, thinking: {thinking1B.content}; answer: {answer1B.content}")
        possible_answers_1B.append(answer1B.content)
        thinkingmapping_1B[answer1B.content] = thinking1B
        answermapping_1B[answer1B.content] = answer1B
    answer1B_content = Counter(possible_answers_1B).most_common(1)[0][0]
    thinking1B = thinkingmapping_1B[answer1B_content]
    answer1B = answermapping_1B[answer1B_content]
    sub_tasks.append(f"Sub-task 1B output: thinking - {thinking1B.content}; answer - {answer1B.content}")
    subtask_desc1B['response'] = {
        "thinking": thinking1B,
        "answer": answer1B
    }
    logs.append(subtask_desc1B)
    debate_instruction_1C = "Sub-task 1C: Implement a debate stage to challenge the uniqueness of the mode and the validity of the median condition for each candidate list generated in subtask 1B."
    debate_agents_1C = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1C = self.max_round
    all_thinking1C = [[] for _ in range(N_max_1C)]
    all_answer1C = [[] for _ in range(N_max_1C)]
    subtask_desc1C = {
        "subtask_id": "subtask_1C",
        "instruction": debate_instruction_1C,
        "context": ["user query", "thinking of subtask 1B", "answer of subtask 1B"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1C):
        for i, agent in enumerate(debate_agents_1C):
            if r == 0:
                thinking1C, answer1C = await agent([taskInfo, thinking1B, answer1B], debate_instruction_1C, r, is_sub_task=True)
            else:
                input_infos_1C = [taskInfo, thinking1B, answer1B] + all_thinking1C[r-1] + all_answer1C[r-1]
                thinking1C, answer1C = await agent(input_infos_1C, debate_instruction_1C, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating list validity, thinking: {thinking1C.content}; answer: {answer1C.content}")
            all_thinking1C[r].append(thinking1C)
            all_answer1C[r].append(answer1C)
    final_decision_agent_1C = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1C, answer1C = await final_decision_agent_1C([taskInfo] + all_thinking1C[-1] + all_answer1C[-1], "Sub-task 1C: Make final decision on list validity.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding list validity, thinking: {thinking1C.content}; answer: {answer1C.content}")
    sub_tasks.append(f"Sub-task 1C output: thinking - {thinking1C.content}; answer - {answer1C.content}")
    subtask_desc1C['response'] = {
        "thinking": thinking1C,
        "answer": answer1C
    }
    logs.append(subtask_desc1C)
    cot_instruction_2 = "Sub-task 2: Select a validated list from subtask 1C that satisfies all conditions: sum of 30, unique mode of 9, and a median that is a positive integer not in the list."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1C", "answer of subtask 1C"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1C, answer1C], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, selecting validated list, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    cot_reflect_instruction_3 = "Sub-task 3: Calculate the sum of the squares of the elements in the validated list from subtask 2."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1C, answer1C, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1C", "answer of subtask 1C", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, calculating sum of squares, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the sum of squares calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining sum of squares calculation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    debate_instruction_4 = "Sub-task 4: Verify the calculations and ensure that the list configuration and the sum of squares meet all initial conditions and constraints."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying calculations, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on verification.", is_sub_task=True)
    agents.append(f"Final Decision agent, verifying calculations, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs