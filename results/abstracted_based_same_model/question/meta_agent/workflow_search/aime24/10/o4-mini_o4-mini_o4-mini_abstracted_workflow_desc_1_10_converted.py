async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Establish the coordinate system by placing point D at (0,0) and aligning D, E, C, F along the x-axis for collinearity."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, establishing coordinate system, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction = "Sub-task 2: Using the coordinate system from subtask 1, place rectangle ABCD with D=(0,0), C=(16,0), B=(0,16), and A=(107,16)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction, "context": ["user query", "Sub-task 1 response"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, placing rectangle ABCD, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction = "Sub-task 3: Using the coordinate system from subtask 1, place rectangle EFGH with E=(x,0), F=(x+184,0), G=(x+184,17), and H=(x,17)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction, "context": ["user query", "Sub-task 1 response"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, placing rectangle EFGH, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 4: Compute symbolic expressions for lengths AD, DH, HG, AG, AH, and DG in terms of x using coordinates from subtasks 2 and 3."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction, "context": ["user query", "Sub-task 2 response", "Sub-task 3 response"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking_i, answer_i = await agent([taskInfo, thinking2, answer2, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing symbolic lengths, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers.append(answer_i.content)
        thinking_mapping[answer_i.content] = thinking_i
        answer_mapping[answer_i.content] = answer_i
    answer4_content = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinking_mapping[answer4_content]
    answer4 = answer_mapping[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction = "Sub-task 5: Apply Ptolemy’s theorem to quadrilateral A-D-H-G to form equation AD*HG + DH*AG = AH*DG."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction, "context": ["user query", "Sub-task 4 response"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent([taskInfo, thinking4, answer4], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, applying Ptolemy’s theorem, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction = "Sub-task 6: Solve the Ptolemy equation for x to obtain all candidate roots."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking5, answer5]
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_reflect_instruction, "context": ["user query", "Sub-task 5 response"], "agent_collaboration": "Reflexion"}
    thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, initial solving for x, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking6, answer6], "Please review the solution for x and provide its validity and limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback round {i}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refined solution round {i+1}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction = "Sub-task 7: Select the valid root between 0 and 16 from the candidate roots for x and assign it as E_x."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_instruction, "context": ["user query", "Sub-task 6 response"], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent([taskInfo, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, selecting valid root for x, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction = "Sub-task 8: Confirm the x-coordinate of C from subtask 2, which should be 16, for use in later CE computations."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_instruction, "context": ["user query", "Sub-task 2 response"], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, confirming C_x, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction = "Sub-task 9: Compute CE1 as the absolute difference |C_x - E_x| using C_x from subtask 8 and E_x from subtask 7."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {"subtask_id": "subtask_9", "instruction": cot_instruction, "context": ["user query", "Sub-task 7 response", "Sub-task 8 response"], "agent_collaboration": "CoT"}
    thinking9, answer9 = await cot_agent([taskInfo, thinking7, answer7, thinking8, answer8], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing CE1, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    cot_instruction = "Sub-task 10: Independently compute CE2 using the distance formula sqrt((C_x - E_x)^2 + (0-0)^2) with C_x from subtask 8 and E_x from subtask 7."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {"subtask_id": "subtask_10", "instruction": cot_instruction, "context": ["user query", "Sub-task 7 response", "Sub-task 8 response"], "agent_collaboration": "CoT"}
    thinking10, answer10 = await cot_agent([taskInfo, thinking7, answer7, thinking8, answer8], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing CE2, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    cot_instruction = "Sub-task 11: Compare CE1 and CE2 to ensure they match; if they differ, indicate a consistency error."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc11 = {"subtask_id": "subtask_11", "instruction": cot_instruction, "context": ["user query", "Sub-task 9 response", "Sub-task 10 response"], "agent_collaboration": "CoT"}
    thinking11, answer11 = await cot_agent([taskInfo, thinking9, answer9, thinking10, answer10], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, comparing CE1 and CE2, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])

    cot_instruction = "Sub-task 12: After confirming consistency, return the integer value of CE as CE1."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc12 = {"subtask_id": "subtask_12", "instruction": cot_instruction, "context": ["user query", "Sub-task 9 response", "Sub-task 11 response"], "agent_collaboration": "CoT"}
    thinking12, answer12 = await cot_agent([taskInfo, thinking9, answer9, thinking11, answer11], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, finalizing CE value, thinking: {thinking12.content}; answer: {answer12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    subtask_desc12['response'] = {"thinking": thinking12, "answer": answer12}
    logs.append(subtask_desc12)
    print("Step 12: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking12, answer12, sub_tasks, agents)
    return final_answer, logs