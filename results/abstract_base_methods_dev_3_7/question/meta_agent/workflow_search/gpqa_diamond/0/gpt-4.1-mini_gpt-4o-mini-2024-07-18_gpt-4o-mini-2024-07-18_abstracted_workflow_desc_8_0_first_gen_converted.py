async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction_1 = "Sub-task 1: Calculate the energy uncertainty (ΔE) for the first quantum state with lifetime 10^-9 sec using the energy-time uncertainty relation ΔE ≈ ħ / (2 * lifetime)."
    N = self.max_sc
    cot_agents1, thinking1, answer1, subtask_desc1, list_thinking1, list_answer1 = await self.sc_cot(subtask_id="subtask_1", cot_sc_instruction=cot_sc_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=N)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    for idx, key in enumerate(list_thinking1):
        agents.append(f"CoT-SC agent {cot_agents1[idx].id}, calculating ΔE for first quantum state, thinking: {list_thinking1[key]}; answer: {list_answer1[key]}")
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = "Sub-task 2: Calculate the energy uncertainty (ΔE) for the second quantum state with lifetime 10^-8 sec using the energy-time uncertainty relation ΔE ≈ ħ / (2 * lifetime)."
    cot_agents2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=N)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx, key in enumerate(list_thinking2):
        agents.append(f"CoT-SC agent {cot_agents2[idx].id}, calculating ΔE for second quantum state, thinking: {list_thinking2[key]}; answer: {list_answer2[key]}")
    logs.append(subtask_desc2)
    cot_sc_instruction_3 = "Sub-task 3: Determine the minimum energy difference required to clearly distinguish the two energy levels by taking the larger of the two energy uncertainties calculated in subtask_1 and subtask_2."
    cot_agents3, thinking3, answer3, subtask_desc3, list_thinking3, list_answer3 = await self.sc_cot(subtask_id="subtask_3", cot_sc_instruction=cot_sc_instruction_3, input_list=[taskInfo, thinking1, answer1, thinking2, answer2], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], n_repeat=N)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    for idx, key in enumerate(list_thinking3):
        agents.append(f"CoT-SC agent {cot_agents3[idx].id}, determining minimum energy difference, thinking: {list_thinking3[key]}; answer: {list_answer3[key]}")
    logs.append(subtask_desc3)
    debate_instruction_4 = "Sub-task 4: Compare the given multiple-choice energy difference options with the minimum required energy difference from subtask_3 and select the option that is greater than or equal to this minimum to ensure clear resolution."
    final_decision_instruction_4 = "Sub-task 4: Make final decision on the correct energy difference option that can clearly resolve the two energy levels."
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "input": [taskInfo, thinking3, answer3],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_4 = {
        "instruction": final_decision_instruction_4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents4, final_decision_agent4, thinking4, answer4, subtask_desc4, list_thinking4, list_answer4 = await self.debate(subtask_id="subtask_4", debate_desc=debate_desc_4, final_decision_desc=final_decision_desc_4, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents4):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing options and selecting correct energy difference, thinking: {list_thinking4[round][idx].content}; answer: {list_answer4[round][idx].content}")
    agents.append(f"Final Decision agent, selecting final energy difference option, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs