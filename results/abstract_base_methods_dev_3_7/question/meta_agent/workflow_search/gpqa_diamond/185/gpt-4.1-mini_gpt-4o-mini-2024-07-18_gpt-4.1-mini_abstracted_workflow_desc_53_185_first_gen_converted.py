async def forward_185(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Analyze and interpret the structure and stereochemistry of (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene to understand the starting material for the Cope rearrangement."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction1 = "Subtask 1: Based on the SC-CoT outputs, refine the analysis of the structure and stereochemistry of the starting material."
    critic_instruction1 = "Please review the refined analysis and provide feedback on any limitations or errors."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1, 'input': [taskInfo] + results1['list_thinking'] + results1['list_answer'], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query"] + results1['list_thinking'] + results1['list_answer']
    }
    critic_desc1 = {
        'instruction': critic_instruction1, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results_reflex1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results1['cot_agent']]}, analyzing structure and stereochemistry, thinking: {results1['list_thinking'][0]}; answer: {results1['list_answer'][0]}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results_reflex1['cot_agent'].id}, refining analysis, thinking: {results_reflex1['list_thinking'][i]}; answer: {results_reflex1['list_answer'][i]}")
        agents.append(f"Critic agent {results_reflex1['critic_agent'].id}, feedback, thinking: {results_reflex1['list_feedback'][i]}; answer: {results_reflex1['list_correct'][i]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results_reflex1['thinking'].content}; answer - {results_reflex1['answer'].content}")
    logs.append(results_reflex1['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Explain the Cope rearrangement mechanism, focusing on how it applies to the given bicyclic azabicyclo compound and the expected stereochemical changes, based on Subtask 1 output."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction2,
        input_list=[taskInfo, results_reflex1['thinking'], results_reflex1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction2 = "Subtask 2: Based on the SC-CoT outputs, refine the explanation of the Cope rearrangement mechanism and stereochemical changes."
    critic_instruction2 = "Please review the refined mechanism explanation and provide feedback on any limitations or errors."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2, 'input': [taskInfo, results_reflex1['thinking'], results_reflex1['answer']] + results2['list_thinking'] + results2['list_answer'], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1"] + results2['list_thinking'] + results2['list_answer']
    }
    critic_desc2 = {
        'instruction': critic_instruction2, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results_reflex2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results2['cot_agent']]}, explaining Cope rearrangement mechanism, thinking: {results2['list_thinking'][0]}; answer: {results2['list_answer'][0]}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results_reflex2['cot_agent'].id}, refining mechanism explanation, thinking: {results_reflex2['list_thinking'][i]}; answer: {results_reflex2['list_answer'][i]}")
        agents.append(f"Critic agent {results_reflex2['critic_agent'].id}, feedback, thinking: {results_reflex2['list_feedback'][i]}; answer: {results_reflex2['list_correct'][i]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results_reflex2['thinking'].content}; answer - {results_reflex2['answer'].content}")
    logs.append(results_reflex2['subtask_desc'])

    cot_instruction3 = "Subtask 3: Predict the product structure resulting from the Cope rearrangement of the given compound, considering stereochemistry and ring transformations, based on Subtask 2 output."
    cot_reflect_instruction3 = "Subtask 3: Refine the predicted product structure considering all stereochemical and mechanistic insights."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3, 'input': [taskInfo, results_reflex2['thinking'], results_reflex2['answer']], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_instruction3 = "Please review the predicted product structure and provide feedback on its validity and stereochemical correctness."
    critic_desc3 = {
        'instruction': critic_instruction3, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results3_cot = await self.cot(
        subtask_id="subtask_3",
        cot_instruction=cot_instruction3,
        input_list=[taskInfo, results_reflex2['thinking'], results_reflex2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "thinking of subtask 2", "answer of subtask 2"]
    )
    results3_reflex = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"CoT agent {results3_cot['cot_agent'].id}, predicting product structure, thinking: {results3_cot['thinking'].content}; answer: {results3_cot['answer'].content}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results3_reflex['cot_agent'].id}, refining product prediction, thinking: {results3_reflex['list_thinking'][i]}; answer: {results3_reflex['list_answer'][i]}")
        agents.append(f"Critic agent {results3_reflex['critic_agent'].id}, feedback, thinking: {results3_reflex['list_feedback'][i]}; answer: {results3_reflex['list_correct'][i]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3_reflex['thinking'].content}; answer - {results3_reflex['answer'].content}")
    logs.append(results3_reflex['subtask_desc'])

    cot_instruction4 = "Subtask 4: Match the predicted product structure to the given multiple-choice options and select the correct answer (A, B, C, or D), based on Subtask 3 output."
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_instruction=cot_instruction4,
        input_list=[taskInfo, results3_reflex['thinking'], results3_reflex['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "thinking of subtask 3", "answer of subtask 3"]
    )
    cot_sc_instruction4 = "Subtask 4: Using self-consistency, confirm the correct multiple-choice answer matching the predicted product structure."
    results4_sc = await self.sc_cot(
        subtask_id="subtask_4_sc",
        cot_sc_instruction=cot_sc_instruction4,
        input_list=[taskInfo, results3_reflex['thinking'], results3_reflex['answer'], results4['thinking'], results4['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        n_repeat=self.max_sc
    )
    final_choice = results4_sc['answer'].content.strip()
    if final_choice not in ['A', 'B', 'C', 'D']:
        final_choice = final_choice[0].upper() if final_choice else 'A'
    sub_tasks.append(f"Subtask 4 output: thinking - {results4_sc['thinking'].content}; answer - {final_choice}")
    agents.append(f"CoT and SC_CoT agents selecting correct multiple-choice answer, thinking: {results4_sc['thinking'].content}; answer: {final_choice}")
    logs.append(results4_sc['subtask_desc'])

    final_answer = await self.make_final_answer(results4_sc['thinking'], final_choice, sub_tasks, agents)
    return final_answer, logs
