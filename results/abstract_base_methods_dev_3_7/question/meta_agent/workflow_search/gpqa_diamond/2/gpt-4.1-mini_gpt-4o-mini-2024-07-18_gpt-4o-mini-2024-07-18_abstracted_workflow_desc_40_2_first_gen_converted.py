async def forward_2(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Identify and write down the given spin state vector in the |\uparrow⟩ and |\downarrow⟩ basis, including normalization and coefficients, based on the input superposition 0.5|\uparrow⟩ + sqrt(3)/2|\downarrow⟩."
    cot_agent1, thinking1, answer1, subtask_desc1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent1.id}, identifying spin state vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = "Sub-task 2: Express the operators \sigma_z and \sigma_x in matrix form in the |\uparrow⟩, |\downarrow⟩ basis."
    cot_agents2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx in range(len(list_thinking2)):
        agents.append(f"CoT-SC agent {cot_agents2[idx].id}, expressing operators \sigma_z and \sigma_x, thinking: {list_thinking2[idx]}; answer: {list_answer2[idx]}")
    logs.append(subtask_desc2)

    cot_sc_instruction_3 = "Sub-task 3: Calculate the expectation value ⟨ψ|\sigma_z|ψ⟩ using the state vector from subtask_1 and operator from subtask_2."
    cot_agents3, thinking3, answer3, subtask_desc3, list_thinking3, list_answer3 = await self.sc_cot(subtask_id="subtask_3", cot_sc_instruction=cot_sc_instruction_3, input_list=[taskInfo, thinking1, answer1, thinking2, answer2], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    for idx in range(len(list_thinking3)):
        agents.append(f"CoT-SC agent {cot_agents3[idx].id}, calculating expectation ⟨ψ|\sigma_z|ψ⟩, thinking: {list_thinking3[idx]}; answer: {list_answer3[idx]}")
    logs.append(subtask_desc3)

    cot_sc_instruction_4 = "Sub-task 4: Calculate the expectation value ⟨ψ|\sigma_x|ψ⟩ using the state vector from subtask_1 and operator from subtask_2."
    cot_agents4, thinking4, answer4, subtask_desc4, list_thinking4, list_answer4 = await self.sc_cot(subtask_id="subtask_4", cot_sc_instruction=cot_sc_instruction_4, input_list=[taskInfo, thinking1, answer1, thinking2, answer2], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    for idx in range(len(list_thinking4)):
        agents.append(f"CoT-SC agent {cot_agents4[idx].id}, calculating expectation ⟨ψ|\sigma_x|ψ⟩, thinking: {list_thinking4[idx]}; answer: {list_answer4[idx]}")
    logs.append(subtask_desc4)

    debate_instruction_5 = "Sub-task 5: Combine the expectation values from subtask_3 and subtask_4 with coefficients 10 and 5 respectively to find the total expectation value of the operator 10\sigma_z + 5\sigma_x."
    final_decision_instruction_5 = "Sub-task 5: Make final decision on the combined expectation value."
    debate_desc = {
        "instruction": debate_instruction_5,
        "context": ["user query", thinking3, answer3, thinking4, answer4],
        "input": [taskInfo, thinking3, answer3, thinking4, answer4],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc = {
        "instruction": final_decision_instruction_5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents5, final_decision_agent5, thinking5, answer5, subtask_desc5, list_thinking5, list_answer5 = await self.debate(subtask_id="subtask_5", debate_desc=debate_desc, final_decision_desc=final_decision_desc, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents5):
            agents.append(f"Debate agent {agent.id}, round {round}, combining expectation values and calculating total, thinking: {list_thinking5[round][idx].content}; answer: {list_answer5[round][idx].content}")
    agents.append(f"Final Decision agent, calculating total expectation value, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    logs.append(subtask_desc5)

    cot_instruction_6 = "Sub-task 6: Compare the computed expectation value from subtask_5 with the given multiple-choice options and select the closest value up to one decimal place."
    cot_agent6, thinking6, answer6, subtask_desc6 = await self.cot(subtask_id="subtask_6", cot_instruction=cot_instruction_6, input_list=[taskInfo, thinking5, answer5], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", thinking5.content, answer5.content])
    agents.append(f"CoT agent {cot_agent6.id}, comparing expectation value with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
