async def forward_1(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Determine the chemical structure of product 1 formed by treating trans-cinnamaldehyde with methylmagnesium bromide, considering the nucleophilic addition of the Grignard reagent to the aldehyde."
    cot_agents1, thinking1, answer1, subtask_desc1, list_thinking1, list_answer1 = await self.sc_cot(subtask_id="subtask_1", cot_sc_instruction=cot_sc_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)

    cot_reflect_instruction_1 = "Sub-task 1 Reflexion: Review and refine the chemical structure determination of product 1 based on SC-CoT outputs."
    critic_instruction_1 = "Please review the validity and limitations of the proposed structures for product 1."
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1, 'input': [taskInfo, thinking1, answer1], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    cot_agent1_reflex, critic_agent1_reflex, thinking1_reflex, answer1_reflex, subtask_desc1_reflex, list_feedback1, list_correct1, list_thinking1_reflex, list_answer1_reflex = await self.reflexion(subtask_id="subtask_1_reflex", cot_reflect_desc=cot_reflect_desc_1, critic_desc=critic_desc_1, n_repeat=self.max_round)

    agents.append(f"SC_CoT agents {[agent.id for agent in cot_agents1]}, thinking: {thinking1.content}; answer: {answer1.content}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {cot_agent1_reflex.id}, round {i}, thinking: {list_thinking1_reflex[i].content}; answer: {list_answer1_reflex[i].content}")
        agents.append(f"Critic agent {critic_agent1_reflex.id}, round {i}, feedback: {list_feedback1[i].content}; correction: {list_correct1[i].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_reflex.content}; answer - {answer1_reflex.content}")
    logs.append(subtask_desc1)
    logs.append(subtask_desc1_reflex)

    cot_sc_instruction_2 = "Sub-task 2: Determine the chemical structure of product 2 formed by treating product 1 with pyridinium chlorochromate (PCC), which oxidizes alcohols to aldehydes or ketones."
    cot_agents2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo, thinking1_reflex, answer1_reflex], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)

    cot_reflect_instruction_2 = "Sub-task 2 Reflexion: Review and refine the chemical structure determination of product 2 based on SC-CoT outputs."
    critic_instruction_2 = "Please review the validity and limitations of the proposed structures for product 2."
    cot_reflect_desc_2 = {
        'instruction': cot_reflect_instruction_2, 'input': [taskInfo, thinking1_reflex, answer1_reflex, thinking2, answer2], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc_2 = {
        'instruction': critic_instruction_2, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    cot_agent2_reflex, critic_agent2_reflex, thinking2_reflex, answer2_reflex, subtask_desc2_reflex, list_feedback2, list_correct2, list_thinking2_reflex, list_answer2_reflex = await self.reflexion(subtask_id="subtask_2_reflex", cot_reflect_desc=cot_reflect_desc_2, critic_desc=critic_desc_2, n_repeat=self.max_round)

    agents.append(f"SC_CoT agents {[agent.id for agent in cot_agents2]}, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {cot_agent2_reflex.id}, round {i}, thinking: {list_thinking2_reflex[i].content}; answer: {list_answer2_reflex[i].content}")
        agents.append(f"Critic agent {critic_agent2_reflex.id}, round {i}, feedback: {list_feedback2[i].content}; correction: {list_correct2[i].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_reflex.content}; answer - {answer2_reflex.content}")
    logs.append(subtask_desc2)
    logs.append(subtask_desc2_reflex)

    cot_instruction_3 = "Sub-task 3: Determine the chemical structure of product 3 formed by treating product 2 with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, identifying the type of reaction and resulting structural changes."
    cot_reflect_instruction_3 = "Sub-task 3 Reflexion: Review and refine the chemical structure determination of product 3 based on CoT output."
    cot_reflect_desc_3 = {
        'instruction': cot_reflect_instruction_3, 'input': [taskInfo, thinking2_reflex, answer2_reflex], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_instruction_3 = "Please review the validity and limitations of the proposed structures for product 3."
    critic_desc_3 = {
        'instruction': critic_instruction_3, 'output': ["feedback", "correct"], 'temperature': 0.0
    }

    cot_agent3, critic_agent3, thinking3, answer3, subtask_desc3, list_feedback3, list_correct3, list_thinking3, list_answer3 = await self.reflexion(subtask_id="subtask_3", cot_reflect_desc=cot_reflect_desc_3, critic_desc=critic_desc_3, n_repeat=self.max_round)

    agents.append(f"CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent3.id}, round {i}, feedback: {list_feedback3[i].content}; correction: {list_correct3[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, round {i}, thinking: {list_thinking3[i].content}; answer: {list_answer3[i].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)

    cot_instruction_4 = "Sub-task 4: Count the number of carbon atoms in product 3 based on the deduced chemical structure from subtask 3, and select the correct multiple-choice answer (A, B, C, or D) corresponding to the carbon count."
    cot_sc_instruction_4 = "Sub-task 4: Using self-consistency, count carbon atoms in product 3 and select the correct answer choice from the given options."

    cot_agents4, thinking4, answer4, subtask_desc4, list_thinking4, list_answer4 = await self.sc_cot(subtask_id="subtask_4", cot_sc_instruction=cot_sc_instruction_4, input_list=[taskInfo, thinking3, answer3], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 3", "answer of subtask 3"], n_repeat=self.max_sc)

    agents.append(f"CoT-SC agents {[agent.id for agent in cot_agents4]}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs