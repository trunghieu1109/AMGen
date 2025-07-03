async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    molecules = [
        "quinuclidine",
        "triisopropyl borate",
        "benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone",
        "triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone"
    ]
    # Stage 1 - Subtask 1: Identify and gather detailed structural information for each molecule
    cot_instruction_1 = "Sub-task 1: Identify and gather detailed structural information for each molecule: quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. This includes molecular geometry, connectivity, and known symmetry features if available."
    cot_agent_1, thinking1, answer1, subtask_desc1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent_1.id}, gathering structural info, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = "Sub-task 2: Analyze the symmetry elements present in each molecule based on their structural information, focusing on identifying rotational axes, mirror planes, and improper rotation axes relevant to C3h symmetry."
    cot_agents_2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo, thinking1, answer1], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx, key in enumerate(list_thinking2):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, analyzing symmetry elements, thinking: {list_thinking2[key]}; answer: {list_answer2[key]}")
    logs.append(subtask_desc2)
    debate_instruction_3 = "Sub-task 3: Determine the point group symmetry of each molecule using the identified symmetry elements, specifically checking if any molecule belongs to the C3h point group."
    final_decision_instruction_3 = "Sub-task 3: Make final decision on the point group symmetry determination for each molecule."
    debate_desc_3 = {
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "input": [taskInfo, thinking2, answer2],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_3 = {
        "instruction": final_decision_instruction_3,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents_3, final_decision_agent_3, thinking3, answer3, subtask_desc3, list_thinking3, list_answer3 = await self.debate(subtask_id="subtask_3", debate_desc=debate_desc_3, final_decision_desc=final_decision_desc_3, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_3):
            agents.append(f"Debate agent {agent.id}, round {round}, determining point group symmetry, thinking: {list_thinking3[round][idx].content}; answer: {list_answer3[round][idx].content}")
    agents.append(f"Final Decision agent {final_decision_agent_3.id}, finalizing point group symmetry, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)
    cot_instruction_4 = "Sub-task 4: Compare the point group results for all molecules and select the molecule that has C3h symmetry. Map this molecule to the corresponding multiple-choice letter (A, B, C, or D) as per the query's choices."
    cot_agent_4, thinking4, answer4, subtask_desc4 = await self.cot(subtask_id="subtask_4", cot_instruction=cot_instruction_4, input_list=[taskInfo, thinking3, answer3], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", "thinking of subtask 3", "answer of subtask 3"])
    agents.append(f"CoT agent {cot_agent_4.id}, comparing point group results and selecting molecule with C3h symmetry, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs