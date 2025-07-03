async def forward_185(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction_1 = "Subtask 1: Extract and define the detailed structural features and stereochemistry of (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene, including the bicyclic framework, vinyl substituent position, and stereocenters. Now reflect: specify whether the vinyl substituent is endo or exo relative to the bicyclic ridge, and describe the ring conformations (axial or equatorial)."
    critic_instruction_1 = "Please review the stereochemical details extracted and provide any missing or ambiguous information."
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1, 'input': [taskInfo], 'output': ['thinking', 'answer'],
        'temperature': 0.0, 'context': ['user query']
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1, 'output': ['feedback', 'correct'], 'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id='subtask_1',
        cot_reflect_desc=cot_reflect_desc_1,
        critic_desc=critic_desc_1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, extracting detailed stereochemistry, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback round {i}, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining stereochemical details, thinking: {results1['list_thinking'][i+1].content}; answer: {results1['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction_2a = "Subtask 2a: Map each atom from the starting material (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene to the six-membered Cope transition state, showing all participating atoms and bonds, including the nitrogen's influence."
    results2a = await self.sc_cot(
        subtask_id='subtask_2a',
        cot_sc_instruction=cot_sc_instruction_2a,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.5,
        context=['user query', 'stereochemical details from subtask 1'],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC-CoT agent {results2a['cot_agent'][idx].id}, atom mapping and transition state analysis, thinking: {results2a['list_thinking'][idx]}; answer: {results2a['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2a output: thinking - {results2a['thinking'].content}; answer - {results2a['answer'].content}")
    logs.append(results2a['subtask_desc'])

    cot_sc_instruction_2b = "Subtask 2b: Enumerate all possible Cope rearrangement products with stereochemical implications based on atom mapping, then vote on the most probable product considering nitrogen's effect and stereochemistry."
    results2b = await self.sc_cot(
        subtask_id='subtask_2b',
        cot_sc_instruction=cot_sc_instruction_2b,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results2a['thinking'], results2a['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.5,
        context=['user query', 'stereochemical details from subtask 1', 'atom mapping from subtask 2a'],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC-CoT agent {results2b['cot_agent'][idx].id}, enumerating and voting on products, thinking: {results2b['list_thinking'][idx]}; answer: {results2b['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2b output: thinking - {results2b['thinking'].content}; answer - {results2b['answer'].content}")
    logs.append(results2b['subtask_desc'])

    cot_reflect_instruction_3 = "Subtask 3: Redraw the predicted Cope rearrangement product with numbered positions and stereochemical descriptors. Then derive the IUPAC name step by step, and validate it by reverse-mapping the name onto the structure to ensure consistency."
    critic_instruction_3 = "Please review the IUPAC naming and structural validation for correctness and consistency."
    cot_reflect_desc_3 = {
        'instruction': cot_reflect_instruction_3,
        'input': [taskInfo, results2b['thinking'], results2b['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'predicted product from subtask 2b']
    }
    critic_desc_3 = {
        'instruction': critic_instruction_3,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id='subtask_3',
        cot_reflect_desc=cot_reflect_desc_3,
        critic_desc=critic_desc_3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, naming and validating product, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback round {i}, thinking: {results3['list_feedback'][i].content}; answer: {results3['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining naming and validation, thinking: {results3['list_thinking'][i+1].content}; answer: {results3['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = "Subtask 4: Critically compare the validated predicted product's structure and IUPAC name with each provided multiple-choice option (A, B, C, D). Argue for or against each choice based on detailed structural and stereochemical features to identify the correct product."
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'context': ['user query', results3['thinking'], results3['answer']],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results4 = await self.debate(
        subtask_id='subtask_4',
        debate_desc=debate_desc_4,
        final_decision_desc=None,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing predicted product with choices, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction_5 = "Subtask 5: Finally, review the selected answer from the debate to confirm it matches the verified product and IUPAC name. Return only the letter choice (A, B, C, or D) as the final answer."
    critic_instruction_5 = "Please verify the final answer choice is consistent with all prior validated information."
    cot_reflect_desc_5 = {
        'instruction': cot_reflect_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'debate results from subtask 4']
    }
    critic_desc_5 = {
        'instruction': critic_instruction_5,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id='subtask_5',
        cot_reflect_desc=cot_reflect_desc_5,
        critic_desc=critic_desc_5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, final answer validation, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback round {i}, thinking: {results5['list_feedback'][i].content}; answer: {results5['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][i+1].content}; answer: {results5['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs