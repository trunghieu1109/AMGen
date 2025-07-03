async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction_1 = "Sub-task 1: Calculate the energy uncertainty (ΔE) for the first quantum state with lifetime 10^-9 sec using ΔE ≈ ħ / (2 * lifetime)."
    cot_agents1, thinking1, answer1, subtask_desc1, list_thinking1, list_answer1 = await self.sc_cot(subtask_id="subtask_1", cot_sc_instruction=cot_sc_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    cot_reflect_instruction_1 = "Sub-task 1 Reflexion: Review the computed ΔE for the first quantum state. Identify any numerical approximations, unit-conversion steps, or assumptions that could be improved. Provide only a concise critique—do not select an answer choice."
    critic_instruction_1 = "Please review the computed energy uncertainty ΔE for the first quantum state and provide its limitations focusing on numerical precision, assumptions, and units."
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1, 'input': [taskInfo, thinking1, answer1], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    cot_agent1, critic_agent1, thinking1_ref, answer1_ref, subtask_desc1_ref, list_feedback1, list_correct1, list_thinking1_ref, list_answer1_ref = await self.reflexion(subtask_id="subtask_1_reflexion", cot_reflect_desc=cot_reflect_desc_1, critic_desc=critic_desc_1, n_repeat=self.max_round)
    agents.append(f"SC_CoT agents {[agent.id for agent in cot_agents1]}, calculating ΔE for first quantum state, thinking: {thinking1.content}; answer: {answer1.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent1.id}, feedback round {i}, thinking: {list_feedback1[i].content}; answer: {list_correct1[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent1.id}, refining round {i}, thinking: {list_thinking1_ref[i+1].content}; answer: {list_answer1_ref[i+1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_ref.content}; answer - {answer1_ref.content}")
    logs.append(subtask_desc1_ref)
    cot_sc_instruction_2 = "Sub-task 2: Calculate the energy uncertainty (ΔE) for the second quantum state with lifetime 10^-8 sec using ΔE ≈ ħ / (2 * lifetime)."
    cot_agents2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    cot_reflect_instruction_2 = "Sub-task 2 Reflexion: Review the computed ΔE for the second quantum state. Identify any numerical approximations, unit-conversion steps, or assumptions that could be improved. Provide only a concise critique."
    critic_instruction_2 = "Please review the computed energy uncertainty ΔE for the second quantum state and provide its limitations focusing on numerical precision, assumptions, and units."
    cot_reflect_desc_2 = {
        'instruction': cot_reflect_instruction_2, 'input': [taskInfo, thinking2, answer2], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc_2 = {
        'instruction': critic_instruction_2, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    cot_agent2, critic_agent2, thinking2_ref, answer2_ref, subtask_desc2_ref, list_feedback2, list_correct2, list_thinking2_ref, list_answer2_ref = await self.reflexion(subtask_id="subtask_2_reflexion", cot_reflect_desc=cot_reflect_desc_2, critic_desc=critic_desc_2, n_repeat=self.max_round)
    agents.append(f"SC_CoT agents {[agent.id for agent in cot_agents2]}, calculating ΔE for second quantum state, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent2.id}, feedback round {i}, thinking: {list_feedback2[i].content}; answer: {list_correct2[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refining round {i}, thinking: {list_thinking2_ref[i+1].content}; answer: {list_answer2_ref[i+1].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_ref.content}; answer - {answer2_ref.content}")
    logs.append(subtask_desc2_ref)
    cot_instruction_3 = "Sub-task 3: Compare the energy uncertainties ΔE₁ and ΔE₂ calculated in subtasks 1 and 2. Show the comparison, state which is larger, and determine the minimum energy difference required to clearly resolve the two energy levels, which must be at least the larger uncertainty."
    cot_agent3, thinking3, answer3, subtask_desc3 = await self.cot(subtask_id="subtask_3", cot_instruction=cot_instruction_3, input_list=[taskInfo, thinking1_ref, answer1_ref, thinking2_ref, answer2_ref], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", "thinking and answer of subtask 1", "thinking and answer of subtask 2"])
    agents.append(f"CoT agent {cot_agent3.id}, comparing ΔE values and determining minimum resolvable energy difference, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)
    cot_reflect_instruction_4 = "Sub-task 4 Reflexion: Review the selection of the energy difference option from the given choices (10^-9 eV, 10^-11 eV, 10^-8 eV, 10^-4 eV) that is equal to or greater than the minimum resolvable energy difference. Confirm that the selected choice’s numeric value indeed exceeds the threshold by at least an order of magnitude. Provide feedback for refinement."
    critic_instruction_4 = "Please review the selected energy difference option and confirm its correctness and adequacy compared to the minimum resolvable energy difference. Provide any limitations or corrections needed."
    cot_reflect_desc_4 = {
        'instruction': cot_reflect_instruction_4, 'input': [taskInfo, thinking3, answer3], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc_4 = {
        'instruction': critic_instruction_4, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    cot_agent4, critic_agent4, thinking4, answer4, subtask_desc4, list_feedback4, list_correct4, list_thinking4, list_answer4 = await self.reflexion(subtask_id="subtask_4", cot_reflect_desc=cot_reflect_desc_4, critic_desc=critic_desc_4, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, selecting energy difference option, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent4.id}, feedback round {i}, thinking: {list_feedback4[i].content}; answer: {list_correct4[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, refining round {i}, thinking: {list_thinking4[i+1].content}; answer: {list_answer4[i+1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs