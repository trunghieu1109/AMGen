async def forward_1(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Determine the chemical structure of product 1 formed by treating trans-cinnamaldehyde with methylmagnesium bromide, considering the reaction mechanism and structural changes."
    cot_agents_1, thinking1, answer1, subtask_desc1, list_thinking1, list_answer1 = await self.sc_cot(subtask_id="subtask_1", cot_sc_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    for idx, key in enumerate(list_thinking1):
        agents.append(f"CoT-SC agent {cot_agents_1[idx].id}, determining product 1 structure, thinking: {list_thinking1[key]}; answer: {list_answer1[key]}")
    logs.append(subtask_desc1)

    cot_instruction_2 = "Sub-task 2: Determine the chemical structure of product 2 formed by treating product 1 with pyridinium chlorochromate (PCC), considering the oxidation reaction and resulting structural changes."
    cot_agents_2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_instruction_2, input_list=[taskInfo, thinking1, answer1], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx, key in enumerate(list_thinking2):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, determining product 2 structure, thinking: {list_thinking2[key]}; answer: {list_answer2[key]}")
    logs.append(subtask_desc2)

    cot_instruction_3 = "Sub-task 3: Determine the chemical structure of product 3 formed by treating product 2 with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, analyzing the reaction type and structural modifications."
    cot_agents_3, thinking3, answer3, subtask_desc3, list_thinking3, list_answer3 = await self.sc_cot(subtask_id="subtask_3", cot_sc_instruction=cot_instruction_3, input_list=[taskInfo, thinking2, answer2], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 2", "answer of subtask 2"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    for idx, key in enumerate(list_thinking3):
        agents.append(f"CoT-SC agent {cot_agents_3[idx].id}, determining product 3 structure, thinking: {list_thinking3[key]}; answer: {list_answer3[key]}")
    logs.append(subtask_desc3)

    debate_instruction_4 = "Sub-task 4: Count the number of carbon atoms in product 3 based on the determined chemical structure from subtask 3, and select the correct multiple-choice answer accordingly."
    final_decision_instruction_4 = "Sub-task 4: Make final decision on the number of carbon atoms in product 3 and select the correct choice."
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "input": [taskInfo, thinking3, answer3],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_4 = {
        "instruction": final_decision_instruction_4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents_4, final_decision_agent_4, thinking4, answer4, subtask_desc4, list_thinking4, list_answer4 = await self.debate(subtask_id="subtask_4", debate_desc=debate_desc_4, final_decision_desc=final_decision_desc_4, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_4):
            agents.append(f"Debate agent {agent.id}, round {round}, counting carbons and selecting answer, thinking: {list_thinking4[round][idx].content}; answer: {list_answer4[round][idx].content}")
    agents.append(f"Final Decision agent, finalizing carbon count and choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs