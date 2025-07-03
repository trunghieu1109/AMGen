async def forward_0(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Identify and summarize the molecular structure and known symmetry elements of each molecule: quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone, based on the given query."
    cot_agent_1, thinking1, answer1, subtask_desc1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent_1.id}, identifying and summarizing molecular structures and symmetry elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)

    cot_instruction_2 = "Sub-task 2: Analyze the symmetry elements identified in Sub-task 1 to determine if each molecule possesses C3h symmetry, focusing on the presence of a C3 rotation axis combined with a horizontal mirror plane."
    cot_agent_2, thinking2, answer2, subtask_desc2 = await self.cot(subtask_id="subtask_2", cot_instruction=cot_instruction_2, input_list=[taskInfo, thinking1, answer1], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", "thinking of subtask 1", "answer of subtask 1"])
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing symmetry elements for C3h symmetry, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(subtask_desc2)

    cot_instruction_3 = "Sub-task 3: Compare the symmetry analysis results from Sub-task 2 and select the molecule that exhibits C3h symmetry among the four choices."
    cot_agent_3, thinking3, answer3, subtask_desc3 = await self.cot(subtask_id="subtask_3", cot_instruction=cot_instruction_3, input_list=[taskInfo, thinking2, answer2], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", "thinking of subtask 2", "answer of subtask 2"])
    agents.append(f"CoT agent {cot_agent_3.id}, comparing symmetry analysis results and selecting molecule with C3h symmetry, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)

    cot_instruction_4 = "Sub-task 4: Format the final answer as a single letter choice (A, B, C, or D) corresponding to the molecule with C3h symmetry, as requested by the query."
    cot_agent_4, thinking4, answer4, subtask_desc4 = await self.cot(subtask_id="subtask_4", cot_instruction=cot_instruction_4, input_list=[taskInfo, thinking3, answer3], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", "thinking of subtask 3", "answer of subtask 3"])
    agents.append(f"CoT agent {cot_agent_4.id}, formatting final answer as single letter choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs