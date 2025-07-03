async def forward_197(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = (
        "Sub-task 1: Write out all equilibrium expressions for cobalt(II) thiocyanato complexes Co(SCN)+, Co(SCN)2, Co(SCN)3-, Co(SCN)4^2-. "
        "Calculate the speciation denominator D = 1 + β1[L] + β2[L]^2 + β3[L]^3 + β4[L]^4 using β1=9, β2=40, β3=63, β4=16, [L]=0.1 M, and total cobalt concentration C_total=0.01 M. "
        "Then calculate each complex concentration explicitly as [Co(SCN)_n] = (β_n * [L]^n) / D * C_total with step-by-step numeric substitution and intermediate values. "
        "Finally, verify that the sum of all complex concentrations does not exceed total cobalt concentration."
    )
    critic_instruction1 = (
        "Please review the calculations of all cobalt(II) thiocyanato complex concentrations, check numeric correctness, and verify physical feasibility including total concentration sum."
    )
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ['user query']
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id='subtask_1',
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, calculating cobalt(II) thiocyanato complex concentrations, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results1['critic_agent'].id}, providing feedback, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining calculations, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = (
        "Sub-task 2: Using the explicit numeric concentrations of all cobalt(II) thiocyanato complexes from Sub-task 1, "
        "calculate the concentration of free (uncomplexed) Co(II) ions by subtracting the sum of all complex concentrations from the total cobalt concentration (0.01 M). "
        "Show all numeric steps clearly."
    )
    results2 = await self.cot(
        subtask_id='subtask_2',
        cot_instruction=cot_instruction2,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.0,
        context=['user query', 'thinking of subtask 1', 'answer of subtask 1']
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, calculating free Co(II) concentration, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = (
        "Sub-task 3: Sum the numeric concentrations of free Co(II) and all cobalt(II) thiocyanato complexes from Sub-tasks 1 and 2. "
        "Compare the total sum to the initial total cobalt concentration (0.01 M), calculate the percent deviation, and report any discrepancy. "
        "Verify the total cobalt balance explicitly."
    )
    critic_instruction3 = (
        "Please review the total cobalt balance verification, check numeric accuracy, and provide feedback on any deviations or errors."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2']
    }
    critic_desc3 = {
        'instruction': critic_instruction3,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id='subtask_3',
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, verifying total cobalt balance, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results3['critic_agent'].id}, providing feedback, thinking: {results3['list_feedback'][i].content}; answer: {results3['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining total cobalt balance verification, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_sc_instruction4 = (
        "Sub-task 4: Calculate the fraction α2 = [Co(SCN)2]/C_total = (β2 * [L]^2) / D from the speciation denominator D obtained in Sub-task 1. "
        "Compute the percentage of the blue dithiocyanato cobalt(II) complex as 100 * α2, showing all intermediate numeric steps explicitly. "
        "Perform multiple reasoning paths to verify the accuracy and consistency of the percentage calculation."
    )
    results4 = await self.sc_cot(
        subtask_id='subtask_4',
        cot_sc_instruction=cot_sc_instruction4,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.5,
        context=['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 3', 'answer of subtask 3'],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, calculating percentage of blue dithiocyanato complex, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
