async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Identify and write down the given spin state vector in the |\uparrow\rangle and |\downarrow\rangle basis, including normalization and coefficients, based on the input superposition 0.5|\uparrow\rangle + sqrt(3)/2|\downarrow\rangle."
    cot_agent1, thinking1, answer1, subtask_desc1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent1.id}, identifying spin state vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)

    cot_instruction_2 = "Sub-task 2: Express the operators \sigma_z and \sigma_x in matrix form in the |\uparrow\rangle, |\downarrow\rangle basis."
    cot_agent2, thinking2, answer2, subtask_desc2 = await self.cot(subtask_id="subtask_2", cot_instruction=cot_instruction_2, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user query")
    agents.append(f"CoT agent {cot_agent2.id}, expressing operators \sigma_z and \sigma_x, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(subtask_desc2)

    cot_instruction_3 = "Sub-task 3: Calculate the expectation value \u27e8\u03c8|\sigma_z|\u03c8\u27e9 using the state vector from subtask_1 and operator from subtask_2."
    cot_agent3, thinking3, answer3, subtask_desc3 = await self.cot(subtask_id="subtask_3", cot_instruction=cot_instruction_3, input_list=[taskInfo, thinking1, answer1, thinking2, answer2], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", thinking1.content, answer1.content, thinking2.content, answer2.content])
    agents.append(f"CoT agent {cot_agent3.id}, calculating expectation \u27e8\u03c8|\sigma_z|\u03c8\u27e9, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)

    cot_reflect_instruction_4 = "Sub-task 4: Calculate the expectation value \u27e8\u03c8|\sigma_x|\u03c8\u27e9 using the state vector from subtask_1 and operator from subtask_2. Do not round \u27e8\u03c8|\sigma_x|\u03c8\u27e9 until the final combined expectation value is computed."
    critic_instruction_4 = "Please review the calculation of \u27e8\u03c8|\sigma_x|\u03c8\u27e9 and ensure no premature rounding is done."
    cot_reflect_desc_4 = {
        'instruction': cot_reflect_instruction_4,
        'input': [taskInfo, thinking1, answer1, thinking2, answer2],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content]
    }
    critic_desc_4 = {
        'instruction': critic_instruction_4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent4, critic_agent4, thinking4, answer4, subtask_desc4, list_feedback4, list_correct4, list_thinking4, list_answer4 = await self.reflexion(subtask_id="subtask_4", cot_reflect_desc=cot_reflect_desc_4, critic_desc=critic_desc_4, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, calculating expectation \u27e8\u03c8|\sigma_x|\u03c8\u27e9, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent4.id}, providing feedback, thinking: {list_feedback4[i].content}; answer: {list_correct4[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refining calculation of \u27e8\u03c8|\sigma_x|\u03c8\u27e9, thinking: {list_thinking4[i + 1].content}; answer: {list_answer4[i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    cot_instruction_5 = "Sub-task 5: Combine the unrounded expectation values \u27e8\u03c8|\sigma_z|\u03c8\u27e9 and \u27e8\u03c8|\sigma_x|\u03c8\u27e9, compute 10*\u27e8\u03c8|\sigma_z|\u03c8\u27e9 + 5*\u27e8\u03c8|\sigma_x|\u03c8\u27e9, then round the final result to one decimal place."
    cot_agent5, thinking5, answer5, subtask_desc5 = await self.cot(subtask_id="subtask_5", cot_instruction=cot_instruction_5, input_list=[taskInfo, thinking3, answer3, thinking4, answer4], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", thinking3.content, answer3.content, thinking4.content, answer4.content])
    agents.append(f"CoT agent {cot_agent5.id}, combining expectation values precisely, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    logs.append(subtask_desc5)

    cot_reflect_instruction_6 = "Sub-task 6: Compare the computed expectation value from subtask_5 with the given multiple-choice options and select the closest value up to one decimal place by calculating absolute differences. Verify correctness of the choice."
    critic_instruction_6 = "Please review the comparison and confirm the closest value selection is correct."
    cot_reflect_desc_6 = {
        'instruction': cot_reflect_instruction_6,
        'input': [taskInfo, thinking5, answer5],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", thinking5.content, answer5.content]
    }
    critic_desc_6 = {
        'instruction': critic_instruction_6,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent6, critic_agent6, thinking6, answer6, subtask_desc6, list_feedback6, list_correct6, list_thinking6, list_answer6 = await self.reflexion(subtask_id="subtask_6", cot_reflect_desc=cot_reflect_desc_6, critic_desc=critic_desc_6, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, comparing expectation value with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent6.id}, providing feedback, thinking: {list_feedback6[i].content}; answer: {list_correct6[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refining choice of closest value, thinking: {list_thinking6[i + 1].content}; answer: {list_answer6[i + 1].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs