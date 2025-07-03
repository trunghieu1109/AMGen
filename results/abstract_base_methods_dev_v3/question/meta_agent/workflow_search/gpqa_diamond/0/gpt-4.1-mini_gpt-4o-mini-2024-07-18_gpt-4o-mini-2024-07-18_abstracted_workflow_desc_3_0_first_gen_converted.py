async def forward_0(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Identify and retrieve the molecular structure and geometry of quinuclidine to enable symmetry analysis." 
    cot_agents_1, thinking_1, answer_1, subtask_desc_1, thinkingmapping_1, answermapping_1 = await self.sc_cot(subtask_id="subtask_1", cot_sc_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    for idx, key in enumerate(thinkingmapping_1):
        agents.append(f"CoT-SC agent {cot_agents_1[idx].id}, retrieving quinuclidine structure, thinking: {thinkingmapping_1[key]}; answer: {answermapping_1[key]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    logs.append(subtask_desc_1)

    cot_instruction_2 = "Sub-task 2: Identify and retrieve the molecular structure and geometry of triisopropyl borate to enable symmetry analysis." 
    cot_agents_2, thinking_2, answer_2, subtask_desc_2, thinkingmapping_2, answermapping_2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_instruction_2, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    for idx, key in enumerate(thinkingmapping_2):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, retrieving triisopropyl borate structure, thinking: {thinkingmapping_2[key]}; answer: {answermapping_2[key]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    logs.append(subtask_desc_2)

    cot_instruction_3 = "Sub-task 3: Identify and retrieve the molecular structure and geometry of benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone to enable symmetry analysis." 
    cot_agents_3, thinking_3, answer_3, subtask_desc_3, thinkingmapping_3, answermapping_3 = await self.sc_cot(subtask_id="subtask_3", cot_sc_instruction=cot_instruction_3, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    for idx, key in enumerate(thinkingmapping_3):
        agents.append(f"CoT-SC agent {cot_agents_3[idx].id}, retrieving benzo trifuran structure, thinking: {thinkingmapping_3[key]}; answer: {answermapping_3[key]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    logs.append(subtask_desc_3)

    cot_instruction_4 = "Sub-task 4: Identify and retrieve the molecular structure and geometry of triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone to enable symmetry analysis." 
    cot_agents_4, thinking_4, answer_4, subtask_desc_4, thinkingmapping_4, answermapping_4 = await self.sc_cot(subtask_id="subtask_4", cot_sc_instruction=cot_instruction_4, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    for idx, key in enumerate(thinkingmapping_4):
        agents.append(f"CoT-SC agent {cot_agents_4[idx].id}, retrieving triphenyleno trifuran structure, thinking: {thinkingmapping_4[key]}; answer: {answermapping_4[key]}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    logs.append(subtask_desc_4)

    cot_instruction_5 = "Sub-task 5: Analyze the symmetry elements of quinuclidine's molecular structure to determine if it has C3h symmetry." 
    cot_agents_5, thinking_5, answer_5, subtask_desc_5, thinkingmapping_5, answermapping_5 = await self.sc_cot(subtask_id="subtask_5", cot_sc_instruction=cot_instruction_5, input_list=[thinking_1, answer_1], output_fields=["thinking", "answer"], temperature=0.5, context=["quinuclidine structure analysis"], n_repeat=self.max_sc)
    for idx, key in enumerate(thinkingmapping_5):
        agents.append(f"CoT-SC agent {cot_agents_5[idx].id}, analyzing quinuclidine symmetry, thinking: {thinkingmapping_5[key]}; answer: {answermapping_5[key]}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    logs.append(subtask_desc_5)

    cot_instruction_6 = "Sub-task 6: Analyze the symmetry elements of triisopropyl borate's molecular structure to determine if it has C3h symmetry." 
    cot_agents_6, thinking_6, answer_6, subtask_desc_6, thinkingmapping_6, answermapping_6 = await self.sc_cot(subtask_id="subtask_6", cot_sc_instruction=cot_instruction_6, input_list=[thinking_2, answer_2], output_fields=["thinking", "answer"], temperature=0.5, context=["triisopropyl borate structure analysis"], n_repeat=self.max_sc)
    for idx, key in enumerate(thinkingmapping_6):
        agents.append(f"CoT-SC agent {cot_agents_6[idx].id}, analyzing triisopropyl borate symmetry, thinking: {thinkingmapping_6[key]}; answer: {answermapping_6[key]}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    logs.append(subtask_desc_6)

    cot_instruction_7 = "Sub-task 7: Analyze the symmetry elements of benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone's molecular structure to determine if it has C3h symmetry." 
    cot_agents_7, thinking_7, answer_7, subtask_desc_7, thinkingmapping_7, answermapping_7 = await self.sc_cot(subtask_id="subtask_7", cot_sc_instruction=cot_instruction_7, input_list=[thinking_3, answer_3], output_fields=["thinking", "answer"], temperature=0.5, context=["benzo trifuran structure analysis"], n_repeat=self.max_sc)
    for idx, key in enumerate(thinkingmapping_7):
        agents.append(f"CoT-SC agent {cot_agents_7[idx].id}, analyzing benzo trifuran symmetry, thinking: {thinkingmapping_7[key]}; answer: {answermapping_7[key]}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_7.content}; answer - {answer_7.content}")
    logs.append(subtask_desc_7)

    cot_instruction_8 = "Sub-task 8: Analyze the symmetry elements of triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone's molecular structure to determine if it has C3h symmetry." 
    cot_agents_8, thinking_8, answer_8, subtask_desc_8, thinkingmapping_8, answermapping_8 = await self.sc_cot(subtask_id="subtask_8", cot_sc_instruction=cot_instruction_8, input_list=[thinking_4, answer_4], output_fields=["thinking", "answer"], temperature=0.5, context=["triphenyleno trifuran structure analysis"], n_repeat=self.max_sc)
    for idx, key in enumerate(thinkingmapping_8):
        agents.append(f"CoT-SC agent {cot_agents_8[idx].id}, analyzing triphenyleno trifuran symmetry, thinking: {thinkingmapping_8[key]}; answer: {answermapping_8[key]}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking_8.content}; answer - {answer_8.content}")
    logs.append(subtask_desc_8)

    debate_instruction_9 = "Sub-task 9: Compare the symmetry analysis results of all four molecules and identify which one has C3h symmetry, then select the corresponding multiple-choice answer." 
    final_decision_instruction_9 = "Sub-task 9: Make final decision on which molecule has C3h symmetry based on previous analyses." 
    debate_desc_9 = {
        "instruction": debate_instruction_9,
        "context": ["user query", thinking_5, answer_5, thinking_6, answer_6, thinking_7, answer_7, thinking_8, answer_8],
        "input": [taskInfo, thinking_5, answer_5, thinking_6, answer_6, thinking_7, answer_7, thinking_8, answer_8],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_9 = {
        "instruction": final_decision_instruction_9,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents_9, final_decision_agent_9, thinking_9, answer_9, subtask_desc_9, all_thinking_9, all_answer_9 = await self.debate(subtask_id="subtask_9", debate_desc=debate_desc_9, final_decision_desc=final_decision_desc_9, n_repeat=self.max_round)
    for r in range(self.max_round):
        for idx, agent in enumerate(debate_agents_9):
            agents.append(f"Debate agent {agent.id}, round {r}, comparing symmetry results, thinking: {all_thinking_9[r][idx].content}; answer: {all_answer_9[r][idx].content}")
    agents.append(f"Final Decision agent, selecting molecule with C3h symmetry, thinking: {thinking_9.content}; answer: {answer_9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking_9.content}; answer - {answer_9.content}")
    logs.append(subtask_desc_9)

    final_answer = await self.make_final_answer(thinking_9, answer_9, sub_tasks, agents)
    return final_answer, logs
