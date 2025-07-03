async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Extract and characterize the molecular structures and key symmetry elements of triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone to identify their essential components relevant to symmetry analysis."
    cot_agent_1, thinking1, answer1, subtask_desc1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent_1.id}, extracting molecular structures, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = "Sub-task 2: Analyze the extracted molecular structures to identify and classify their symmetry elements, specifically checking for the presence of C3h symmetry in each molecule, based on the output from Sub-task 1."
    cot_agents2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo, thinking1, answer1], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx, key in enumerate(list_thinking2):
        agents.append(f"CoT-SC agent {cot_agents2[idx].id}, analyzing symmetry elements, thinking: {list_thinking2[key]}; answer: {list_answer2[key]}")
    logs.append(subtask_desc2)
    debate_instruction_3 = "Sub-task 3: Compare the symmetry classifications of all four molecules and determine which one exhibits C3h symmetry, then map this result to the corresponding multiple-choice answer (A, B, C, or D), based on the output of Sub-task 2."
    final_decision_instruction_3 = "Sub-task 3: Make final decision on which molecule has C3h symmetry and provide the mapped multiple-choice answer."
    debate_desc = {
        "instruction": debate_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "input": [taskInfo, thinking2, answer2],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc = {
        "instruction": final_decision_instruction_3,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents3, final_decision_agent3, thinking3, answer3, subtask_desc3, list_thinking3, list_answer3 = await self.debate(subtask_id="subtask_3", debate_desc=debate_desc, final_decision_desc=final_decision_desc, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents3):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing symmetry classifications and mapping to answer, thinking: {list_thinking3[round][idx].content}; answer: {list_answer3[round][idx].content}")
    agents.append(f"Final Decision agent, determining final molecule with C3h symmetry, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs