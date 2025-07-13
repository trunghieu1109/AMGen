async def forward_1(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Identify and define geometric elements
    cot_instruction_1 = "Sub-task 1: Identify and clearly define all geometric elements in the problem, including points A, B, C, D, P, and circle ω. Label segments such as AD, DP, and AP, specifying their order on the secant line AD."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing geometric configuration, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Apply the Power of a Point theorem
    cot_sc_instruction_2 = "Sub-task 2: Apply the Power of a Point theorem correctly, ensuring the correct segment naming and orientation, to relate the segments AD, DP, and AP."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of stage_1_subtask_1", "answer of stage_1_subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, applying Power of a Point theorem, thinking: {thinking2.content}; answer: {answer2.content}")
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

    # Stage 1: Calculate semi-perimeter and area
    cot_instruction_3 = "Sub-task 3: Use the given side lengths of triangle ABC to calculate the semi-perimeter and area of the triangle."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating semi-perimeter and area, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Calculate the circumradius
    cot_instruction_4 = "Sub-task 4: Calculate the circumradius of triangle ABC using the area and semi-perimeter."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of stage_1_subtask_3", "answer of stage_1_subtask_3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating circumradius, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Determine the length of segment AD
    cot_instruction_5 = "Sub-task 5: Determine the length of segment AD using the properties of tangents and the circumradius."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of stage_1_subtask_2", "answer of stage_1_subtask_2", "thinking of stage_2_subtask_1", "answer of stage_2_subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking2, answer2, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, determining length of AD, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Calculate the length of segment AP
    cot_instruction_6 = "Sub-task 6: Calculate the length of segment AP using the relationship derived from the Power of a Point theorem, ensuring all steps are shown."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "stage_2_subtask_3",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of stage_2_subtask_2", "answer of stage_2_subtask_2"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating length of AP, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Stage 3: Express AP in the form of a fraction m/n
    cot_instruction_7 = "Sub-task 7: Express AP in the form of a fraction m/n where m and n are relatively prime, and verify the result through an independent method such as coordinate geometry or dropping perpendiculars."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of stage_2_subtask_3", "answer of stage_2_subtask_3"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, expressing AP as m/n, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    # Stage 3: Perform a sanity check
    cot_instruction_8 = "Sub-task 8: Perform a sanity check by approximating and assessing the reasonableness of all lengths and results before finalizing the answer."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "stage_3_subtask_2",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of stage_3_subtask_1", "answer of stage_3_subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, performing sanity check, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {
        "thinking": thinking8,
        "answer": answer8
    }
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    # Stage 3: Compute the sum m + n
    cot_instruction_9 = "Sub-task 9: Compute the sum m + n."
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "stage_3_subtask_3",
        "instruction": cot_instruction_9,
        "context": ["user query", "thinking of stage_3_subtask_2", "answer of stage_3_subtask_2"],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, computing sum m + n, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {
        "thinking": thinking9,
        "answer": answer9
    }
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs