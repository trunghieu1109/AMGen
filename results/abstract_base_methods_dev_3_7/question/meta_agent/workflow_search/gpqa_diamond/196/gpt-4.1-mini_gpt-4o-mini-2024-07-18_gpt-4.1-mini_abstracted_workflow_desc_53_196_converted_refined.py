async def forward_196(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Interpret the IR spectrum data to identify key functional groups present in compound X, focusing on the broad 3400–2500 cm-1, 1720 cm-1, 1610 cm-1, and 1450 cm-1 peaks. List each IR peak and confirm which functional group it supports, including consideration of alternative groups and justification for the chosen assignment."
    N1 = self.max_sc
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=N1
    )
    cot_reflect_instruction1 = "Subtask 1 Reflexion: Review the IR spectrum interpretation and provide feedback on the identified functional groups, considering possible alternative assignments and justifications."
    critic_instruction1 = "Please review the IR spectrum interpretation and provide its limitations or possible errors."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1, 'input': [taskInfo] + results1['list_thinking'] + results1['list_answer'], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query"] + results1['list_thinking'] + results1['list_answer']
    }
    critic_desc1 = {
        'instruction': critic_instruction1, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results1_reflexion = await self.reflexion(
        subtask_id="subtask_1_reflexion",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results1['cot_agent']]}, interpreting IR spectrum, thinking samples: {results1['list_thinking']}; answers: {results1['list_answer']}")
    agents.append(f"Reflexion agent {results1_reflexion['cot_agent'].id}, reviewing IR interpretation, thinking: {results1_reflexion['list_thinking'][0].content}; answer: {results1_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results1_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results1_reflexion['list_feedback'][i].content}; answer: {results1_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion agent {results1_reflexion['cot_agent'].id}, refining IR interpretation round {i}, thinking: {results1_reflexion['list_thinking'][i+1].content}; answer: {results1_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1_reflexion['thinking'].content}; answer - {results1_reflexion['answer'].content}")
    logs.append(results1_reflexion['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Analyze the 1H NMR data to deduce the structural features of compound X, including the types and environments of hydrogen atoms indicated by chemical shifts and splitting patterns, using the IR interpretation as context. Verify that the total number of protons assigned matches the molecular formula and discuss possible ambiguities or alternative interpretations of the NMR data."
    N2 = self.max_sc
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction2,
        input_list=[taskInfo, results1_reflexion['thinking'], results1_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results1_reflexion['thinking'], results1_reflexion['answer']],
        n_repeat=N2
    )
    cot_reflect_instruction2 = "Subtask 2 Reflexion: Review the NMR structural deduction and provide feedback on the proposed hydrogen environments, splitting patterns, and any ambiguities or alternative interpretations."
    critic_instruction2 = "Please review the NMR analysis and provide its limitations or possible errors."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2, 'input': [taskInfo, results1_reflexion['thinking'], results1_reflexion['answer']] + results2['list_thinking'] + results2['list_answer'], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", results1_reflexion['thinking'], results1_reflexion['answer']] + results2['list_thinking'] + results2['list_answer']
    }
    critic_desc2 = {
        'instruction': critic_instruction2, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results2_reflexion = await self.reflexion(
        subtask_id="subtask_2_reflexion",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results2['cot_agent']]}, analyzing NMR data, thinking samples: {results2['list_thinking']}; answers: {results2['list_answer']}")
    agents.append(f"Reflexion agent {results2_reflexion['cot_agent'].id}, reviewing NMR analysis, thinking: {results2_reflexion['list_thinking'][0].content}; answer: {results2_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results2_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results2_reflexion['list_feedback'][i].content}; answer: {results2_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion agent {results2_reflexion['cot_agent'].id}, refining NMR analysis round {i}, thinking: {results2_reflexion['list_thinking'][i+1].content}; answer: {results2_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2_reflexion['thinking'].content}; answer - {results2_reflexion['answer'].content}")
    logs.append(results2_reflexion['subtask_desc'])

    cot_instruction3 = "Subtask 3: Combine the IR and NMR interpretations to propose the most likely structure of compound X before reaction, ensuring consistency with all spectral data. Explicitly compare each spectral data point with the proposed structure and note any discrepancies or inconsistencies."
    cot_reflect_instruction3 = "Subtask 3 Reflexion: Review the proposed structure for consistency and plausibility based on spectral data, highlighting any contradictions or uncertainties."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3, 'input': [taskInfo, results1_reflexion['thinking'], results1_reflexion['answer'], results2_reflexion['thinking'], results2_reflexion['answer']], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", results1_reflexion['thinking'], results1_reflexion['answer'], results2_reflexion['thinking'], results2_reflexion['answer']]
    }
    critic_instruction3 = "Please review the proposed structure and provide feedback on its validity and any inconsistencies."
    critic_desc3 = {
        'instruction': critic_instruction3, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results3_cot = await self.cot(
        subtask_id="subtask_3_cot",
        cot_instruction=cot_instruction3,
        input_list=[taskInfo, results1_reflexion['thinking'], results1_reflexion['answer'], results2_reflexion['thinking'], results2_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", results1_reflexion['thinking'], results1_reflexion['answer'], results2_reflexion['thinking'], results2_reflexion['answer']]
    )
    results3_reflexion = await self.reflexion(
        subtask_id="subtask_3_reflexion",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"CoT agent {results3_cot['cot_agent'].id}, proposing structure, thinking: {results3_cot['thinking'].content}; answer: {results3_cot['answer'].content}")
    agents.append(f"Reflexion agent {results3_reflexion['cot_agent'].id}, reviewing proposed structure, thinking: {results3_reflexion['list_thinking'][0].content}; answer: {results3_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results3_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results3_reflexion['list_feedback'][i].content}; answer: {results3_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion agent {results3_reflexion['cot_agent'].id}, refining structure round {i}, thinking: {results3_reflexion['list_thinking'][i+1].content}; answer: {results3_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3_reflexion['thinking'].content}; answer - {results3_reflexion['answer'].content}")
    logs.append(results3_reflexion['subtask_desc'])

    cot_instruction4 = "Subtask 4: Detail the reaction mechanism of red phosphorus and HI with carboxylic acids, outlining possible products such as decarboxylation to alkane or conversion to alkyl halide. Predict the most chemically consistent final product structure of compound X after reaction, considering typical reaction pathways and mechanistic steps."
    results4_cot = await self.cot(
        subtask_id="subtask_4_cot",
        cot_instruction=cot_instruction4,
        input_list=[taskInfo, results3_reflexion['thinking'], results3_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", results3_reflexion['thinking'], results3_reflexion['answer']]
    )
    sub_tasks.append(f"Subtask 4 output: thinking - {results4_cot['thinking'].content}; answer - {results4_cot['answer'].content}")
    agents.append(f"CoT agent {results4_cot['cot_agent'].id}, analyzing reaction mechanism and predicting product, thinking: {results4_cot['thinking'].content}; answer: {results4_cot['answer'].content}")
    logs.append(results4_cot['subtask_desc'])

    cot_sc_instruction5 = "Subtask 5: Based on the predicted final product structure from Subtask 4, restate the predicted product name, scan the multiple-choice options (A: 1-(sec-butyl)-4-methylbenzene, B: 4-(sec-butyl)benzoic acid, C: 2-(4-ethylphenyl)propanoic acid, D: 1-isobutyl-4-methylbenzene) to find the best match, and cross-verify the choice with the chemical transformation analysis before finalizing the answer."
    N5 = self.max_sc
    results5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_instruction=cot_sc_instruction5,
        input_list=[taskInfo, results4_cot['thinking'], results4_cot['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results4_cot['thinking'], results4_cot['answer']],
        n_repeat=N5
    )
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    for idx, key in enumerate(results5['list_thinking']):
        agents.append(f"SC-CoT agent {results5['cot_agent'][idx].id}, considering product matching and verification, thinking: {results5['list_thinking'][idx]}; answer: {results5['list_answer'][idx]}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
