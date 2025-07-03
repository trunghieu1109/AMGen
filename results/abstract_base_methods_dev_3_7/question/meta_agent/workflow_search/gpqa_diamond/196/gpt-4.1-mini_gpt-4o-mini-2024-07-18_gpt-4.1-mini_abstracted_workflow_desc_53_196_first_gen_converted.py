async def forward_196(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Interpret the IR spectrum data to identify key functional groups present in compound X, focusing on the broad 3400–2500 cm-1, 1720 cm-1, 1610 cm-1, and 1450 cm-1 peaks."
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
    cot_reflect_instruction1 = "Subtask 1 Reflexion: Review the IR spectrum interpretation and provide feedback on the identified functional groups."
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

    cot_sc_instruction2 = "Subtask 2: Analyze the 1H NMR data to deduce the structural features of compound X, including the types and environments of hydrogen atoms indicated by chemical shifts and splitting patterns, using the IR interpretation as context."
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
    cot_reflect_instruction2 = "Subtask 2 Reflexion: Review the NMR structural deduction and provide feedback on the proposed hydrogen environments and splitting patterns."
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

    cot_instruction3 = "Subtask 3: Combine the IR and NMR interpretations to propose the most likely structure of compound X before reaction, ensuring consistency with all spectral data."
    cot_reflect_instruction3 = "Subtask 3 Reflexion: Review the proposed structure for consistency and plausibility based on spectral data."
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

    cot_instruction4 = "Subtask 4: Analyze the reaction conditions (red phosphorus and HI) to determine the chemical transformation that compound X undergoes, predicting the final product structure."
    cot_reflect_instruction4 = "Subtask 4 Reflexion: Review the predicted chemical transformation and final product structure for accuracy and consistency."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4, 'input': [taskInfo, results3_reflexion['thinking'], results3_reflexion['answer']], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", results3_reflexion['thinking'], results3_reflexion['answer']]
    }
    critic_instruction4 = "Please review the predicted chemical transformation and final product structure and provide feedback on its validity."
    critic_desc4 = {
        'instruction': critic_instruction4, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    results4_cot = await self.cot(
        subtask_id="subtask_4_cot",
        cot_instruction=cot_instruction4,
        input_list=[taskInfo, results3_reflexion['thinking'], results3_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", results3_reflexion['thinking'], results3_reflexion['answer']]
    )
    results4_reflexion = await self.reflexion(
        subtask_id="subtask_4_reflexion",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"CoT agent {results4_cot['cot_agent'].id}, analyzing reaction, thinking: {results4_cot['thinking'].content}; answer: {results4_cot['answer'].content}")
    agents.append(f"Reflexion agent {results4_reflexion['cot_agent'].id}, reviewing reaction analysis, thinking: {results4_reflexion['list_thinking'][0].content}; answer: {results4_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results4_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results4_reflexion['list_feedback'][i].content}; answer: {results4_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion agent {results4_reflexion['cot_agent'].id}, refining reaction analysis round {i}, thinking: {results4_reflexion['list_thinking'][i+1].content}; answer: {results4_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4_reflexion['thinking'].content}; answer - {results4_reflexion['answer'].content}")
    logs.append(results4_reflexion['subtask_desc'])

    debate_instruction5 = "Subtask 5: Based on the output of Subtask 4, match the predicted final product structure with the given multiple-choice options and select the correct answer (A, B, C, or D)."
    final_decision_instruction5 = "Subtask 5: Make final decision on the correct multiple-choice answer for the final product."
    debate_desc5 = {
        "instruction": debate_instruction5,
        "context": ["user query", results4_reflexion['thinking'], results4_reflexion['answer']],
        "input": [taskInfo, results4_reflexion['thinking'], results4_reflexion['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc5 = {
        "instruction": final_decision_instruction5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        final_decision_desc=final_decision_desc5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating final product matching, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
