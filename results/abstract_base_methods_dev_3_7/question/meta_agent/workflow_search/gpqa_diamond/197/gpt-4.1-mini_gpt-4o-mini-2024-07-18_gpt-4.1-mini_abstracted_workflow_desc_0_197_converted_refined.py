async def forward_197(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = "Sub-task 1: Identify all cobalt(II) thiocyanato complex species formed in solution, explicitly including their charge states (e.g., Co(SCN)^+, Co(SCN)_2^0), and write their formation equilibria with given stability constants β1=9, β2=40, β3=63, β4=16. Ensure consistent notation and charge representation throughout."
    critic_instruction1 = "Please review the identified cobalt(II) thiocyanato complex species and their formation equilibria for completeness, correctness, and notation consistency including charge states."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
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
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, round {i}, identifying cobalt(II) thiocyanato complexes, thinking: {results1['list_thinking'][i]}; answer: {results1['list_answer'][i]}")
        agents.append(f"Critic agent {results1['critic_agent'].id}, round {i}, feedback: {results1['list_feedback'][i]}; correctness: {results1['list_correct'][i]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction2 = "Sub-task 2: Write the total cobalt mass balance as the sum of free Co(II) and all complexed species concentrations, and write the total thiocyanate (SCN⁻) ligand mass balance. Formulate equilibrium expressions using the given β values and initial concentrations c(Co) = 10^-2 M and [SCN-] = 0.1 M, ensuring free ligand concentration is not assumed constant."
    critic_instruction2 = "Please review the mass balance and equilibrium expressions for correctness, inclusion of ligand balance, and consistency with Sub-task 1 output."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id='subtask_2',
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, round {i}, formulating mass and ligand balance, thinking: {results2['list_thinking'][i]}; answer: {results2['list_answer'][i]}")
        agents.append(f"Critic agent {results2['critic_agent'].id}, round {i}, feedback: {results2['list_feedback'][i]}; correctness: {results2['list_correct'][i]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = "Sub-task 3: Using the mass balance and equilibrium expressions from Sub-task 2, solve step-by-step the simultaneous equations for equilibrium concentrations of free Co(II), free SCN⁻, and all cobalt thiocyanato complexes (mono-, di-, tri-, and tetrathiocyanato). Show detailed algebraic or numerical steps including intermediate results to ensure transparency and correctness."
    results3 = await self.cot(
        subtask_id='subtask_3',
        cot_instruction=cot_instruction3,
        input_list=[taskInfo, results2['thinking'], results2['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.0,
        context=['user query', 'thinking of subtask 2', 'answer of subtask 2']
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, solving simultaneous mass-balance equations, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_reflect_instruction4 = "Sub-task 4: Verify that the sum of the fractions of all cobalt species equals 100%, and explicitly calculate the fraction of the blue dithiocyanato cobalt(II) complex relative to total cobalt. Confirm consistency and correctness of the fraction calculation based on Sub-task 3 results."
    critic_instruction4 = "Please review the fraction calculations for the dithiocyanato complex and the sum of all species fractions, providing feedback on any inconsistencies or errors."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3']
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id='subtask_4',
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, round {i}, verifying species fractions, thinking: {results4['list_thinking'][i]}; answer: {results4['list_answer'][i]}")
        agents.append(f"Critic agent {results4['critic_agent'].id}, round {i}, feedback: {results4['list_feedback'][i]}; correctness: {results4['list_correct'][i]}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction5 = "Sub-task 5: Convert the fraction of the blue dithiocyanato cobalt(II) complex to a percentage and compare it with the given multiple-choice options (16.9%, 42.3%, 25.6%, 38.1%). Each debate agent should evaluate the closeness of each choice to the computed value and vote for the best match, considering rounding and approximation."
    final_decision_instruction5 = "Sub-task 5: Make the final decision on the correct percentage choice of the blue dithiocyanato cobalt(II) complex among all cobalt species based on the debate votes."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ['user query', results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    final_decision_desc5 = {
        'instruction': final_decision_instruction5,
        'output': ['thinking', 'answer'],
        'temperature': 0.0
    }
    results5 = await self.debate(
        subtask_id='subtask_5',
        debate_desc=debate_desc5,
        final_decision_desc=final_decision_desc5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, evaluating answer choices, thinking: {results5['list_thinking'][round][idx]}; answer: {results5['list_answer'][round][idx]}")
    agents.append(f"Final Decision agent, calculating final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
