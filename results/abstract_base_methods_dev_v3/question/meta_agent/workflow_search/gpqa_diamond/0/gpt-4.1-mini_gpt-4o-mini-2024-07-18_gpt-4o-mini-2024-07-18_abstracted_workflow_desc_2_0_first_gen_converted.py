async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Collect and confirm the molecular structures and known symmetry information for each of the four molecules: triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. Retrieve or verify their 3D geometries and any documented symmetry point groups."
    cot_agent_1, thinking1, answer1, subtask_desc1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent_1.id}, collecting molecular structures and symmetry info, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = "Sub-task 2: Analyze the molecular symmetry elements of each molecule based on their structures from Sub-task 1 to determine their point groups, focusing on identifying the presence or absence of C3h symmetry elements."
    cot_agents_2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo, thinking1, answer1], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx, key in enumerate(list_thinking2):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, analyzing symmetry elements, thinking: {list_thinking2[key]}; answer: {list_answer2[key]}")
    logs.append(subtask_desc2)
    cot_sc_instruction_3 = "Sub-task 3: Compare the determined point groups of all four molecules from Sub-task 2 to the C3h point group and identify which molecule(s) match the C3h symmetry."
    cot_agents_3, thinking3, answer3, subtask_desc3, list_thinking3, list_answer3 = await self.sc_cot(subtask_id="subtask_3", cot_sc_instruction=cot_sc_instruction_3, input_list=[taskInfo, thinking2, answer2], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 2", "answer of subtask 2"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    for idx, key in enumerate(list_thinking3):
        agents.append(f"CoT-SC agent {cot_agents_3[idx].id}, comparing point groups to C3h, thinking: {list_thinking3[key]}; answer: {list_answer3[key]}")
    logs.append(subtask_desc3)
    cot_sc_instruction_4 = "Sub-task 4: Map the molecule identified with C3h symmetry from Sub-task 3 to the corresponding multiple-choice letter (A, B, C, or D) and provide the final answer as required by the query format."
    cot_agents_4, thinking4, answer4, subtask_desc4, list_thinking4, list_answer4 = await self.sc_cot(subtask_id="subtask_4", cot_sc_instruction=cot_sc_instruction_4, input_list=[taskInfo, thinking3, answer3], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 3", "answer of subtask 3"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    for idx, key in enumerate(list_thinking4):
        agents.append(f"CoT-SC agent {cot_agents_4[idx].id}, mapping molecule to choice letter, thinking: {list_thinking4[key]}; answer: {list_answer4[key]}")
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs