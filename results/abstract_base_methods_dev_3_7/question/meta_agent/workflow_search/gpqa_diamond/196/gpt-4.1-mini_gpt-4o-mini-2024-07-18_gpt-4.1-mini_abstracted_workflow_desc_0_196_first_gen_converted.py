async def forward_196(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Analyze the IR spectrum to identify key functional groups present in Compound X based on characteristic absorption bands from the given IR data and context."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results1['list_thinking']):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, analyzing IR spectrum, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction_1 = "Sub-task 1 Reflexion: Review and refine the IR spectrum analysis results to ensure accurate identification of functional groups in Compound X."
    critic_instruction_1 = "Please review the IR spectrum analysis and provide feedback on its accuracy and limitations."
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1_reflexion = await self.reflexion(
        subtask_id="subtask_1_reflexion",
        cot_reflect_desc=cot_reflect_desc_1,
        critic_desc=critic_desc_1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1_reflexion['cot_agent'].id}, refining IR analysis, thinking: {results1_reflexion['list_thinking'][0].content}; answer: {results1_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results1_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results1_reflexion['list_feedback'][i].content}; answer: {results1_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results1_reflexion['cot_agent'].id}, refining IR analysis round {i+1}, thinking: {results1_reflexion['list_thinking'][i+1].content}; answer: {results1_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 1 Reflexion output: thinking - {results1_reflexion['thinking'].content}; answer - {results1_reflexion['answer'].content}")
    logs.append(results1_reflexion['subtask_desc'])

    cot_sc_instruction_2 = "Sub-task 2: Analyze the 1H NMR data to deduce the structural features and proton environments in Compound X based on the given NMR data and context."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction_2,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, analyzing NMR data, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction_2 = "Sub-task 2 Reflexion: Review and refine the NMR data analysis to ensure accurate deduction of structural features and proton environments in Compound X."
    critic_instruction_2 = "Please review the NMR data analysis and provide feedback on its accuracy and limitations."
    cot_reflect_desc_2 = {
        'instruction': cot_reflect_instruction_2,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc_2 = {
        'instruction': critic_instruction_2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2_reflexion = await self.reflexion(
        subtask_id="subtask_2_reflexion",
        cot_reflect_desc=cot_reflect_desc_2,
        critic_desc=critic_desc_2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2_reflexion['cot_agent'].id}, refining NMR analysis, thinking: {results2_reflexion['list_thinking'][0].content}; answer: {results2_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results2_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results2_reflexion['list_feedback'][i].content}; answer: {results2_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results2_reflexion['cot_agent'].id}, refining NMR analysis round {i+1}, thinking: {results2_reflexion['list_thinking'][i+1].content}; answer: {results2_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 2 Reflexion output: thinking - {results2_reflexion['thinking'].content}; answer - {results2_reflexion['answer'].content}")
    logs.append(results2_reflexion['subtask_desc'])

    cot_sc_instruction_3 = "Sub-task 3: Integrate IR and NMR analysis results to propose the most plausible structure or functional groups present in Compound X before reaction."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction_3,
        input_list=[taskInfo, results1_reflexion['thinking'], results1_reflexion['answer'], results2_reflexion['thinking'], results2_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "IR analysis thinking", "IR analysis answer", "NMR analysis thinking", "NMR analysis answer"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, integrating IR and NMR analyses, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_reflect_instruction_3 = "Sub-task 3 Reflexion: Review and refine the integrated structural proposal based on IR and NMR analyses to ensure the most plausible structure before reaction."
    critic_instruction_3 = "Please review the integrated structural proposal and provide feedback on its validity and limitations."
    cot_reflect_desc_3 = {
        'instruction': cot_reflect_instruction_3,
        'input': [taskInfo, results1_reflexion['thinking'], results1_reflexion['answer'], results2_reflexion['thinking'], results2_reflexion['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "IR analysis thinking", "IR analysis answer", "NMR analysis thinking", "NMR analysis answer", "integration thinking", "integration answer"]
    }
    critic_desc_3 = {
        'instruction': critic_instruction_3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3_reflexion = await self.reflexion(
        subtask_id="subtask_3_reflexion",
        cot_reflect_desc=cot_reflect_desc_3,
        critic_desc=critic_desc_3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3_reflexion['cot_agent'].id}, refining integrated structure, thinking: {results3_reflexion['list_thinking'][0].content}; answer: {results3_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results3_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results3_reflexion['list_feedback'][i].content}; answer: {results3_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results3_reflexion['cot_agent'].id}, refining integrated structure round {i+1}, thinking: {results3_reflexion['list_thinking'][i+1].content}; answer: {results3_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 3 Reflexion output: thinking - {results3_reflexion['thinking'].content}; answer - {results3_reflexion['answer'].content}")
    logs.append(results3_reflexion['subtask_desc'])

    cot_sc_instruction_4 = "Sub-task 4: Assess the chemical effect of treating Compound X with red phosphorus and HI, focusing on expected transformations of identified functional groups from the integrated structure."
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_instruction=cot_sc_instruction_4,
        input_list=[taskInfo, results3_reflexion['thinking'], results3_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "integrated structure thinking", "integrated structure answer"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results4['list_thinking']):
        agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, assessing reaction effect, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction_4 = "Sub-task 4 Reflexion: Review and refine the assessment of chemical transformations upon treatment with red phosphorus and HI."
    critic_instruction_4 = "Please review the reaction effect assessment and provide feedback on its accuracy and limitations."
    cot_reflect_desc_4 = {
        'instruction': cot_reflect_instruction_4,
        'input': [taskInfo, results3_reflexion['thinking'], results3_reflexion['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "integrated structure thinking", "integrated structure answer", "reaction effect thinking", "reaction effect answer"]
    }
    critic_desc_4 = {
        'instruction': critic_instruction_4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4_reflexion = await self.reflexion(
        subtask_id="subtask_4_reflexion",
        cot_reflect_desc=cot_reflect_desc_4,
        critic_desc=critic_desc_4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4_reflexion['cot_agent'].id}, refining reaction effect assessment, thinking: {results4_reflexion['list_thinking'][0].content}; answer: {results4_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results4_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results4_reflexion['list_feedback'][i].content}; answer: {results4_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results4_reflexion['cot_agent'].id}, refining reaction effect assessment round {i+1}, thinking: {results4_reflexion['list_thinking'][i+1].content}; answer: {results4_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 4 Reflexion output: thinking - {results4_reflexion['thinking'].content}; answer - {results4_reflexion['answer'].content}")
    logs.append(results4_reflexion['subtask_desc'])

    cot_sc_instruction_5 = "Sub-task 5: Based on the predicted product from the reaction and the structural features deduced, match the final product to one of the given multiple-choice options."
    results5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_instruction=cot_sc_instruction_5,
        input_list=[taskInfo, results4_reflexion['thinking'], results4_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "reaction effect thinking", "reaction effect answer"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results5['list_thinking']):
        agents.append(f"CoT-SC agent {results5['cot_agent'][idx].id}, matching final product to choices, thinking: {results5['list_thinking'][idx]}; answer: {results5['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_reflect_instruction_5 = "Sub-task 5 Reflexion: Review and refine the final product matching to ensure the most accurate choice selection."
    critic_instruction_5 = "Please review the final product matching and provide feedback on its correctness and limitations."
    cot_reflect_desc_5 = {
        'instruction': cot_reflect_instruction_5,
        'input': [taskInfo, results4_reflexion['thinking'], results4_reflexion['answer'], results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "reaction effect thinking", "reaction effect answer", "final matching thinking", "final matching answer"]
    }
    critic_desc_5 = {
        'instruction': critic_instruction_5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5_reflexion = await self.reflexion(
        subtask_id="subtask_5_reflexion",
        cot_reflect_desc=cot_reflect_desc_5,
        critic_desc=critic_desc_5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5_reflexion['cot_agent'].id}, refining final product matching, thinking: {results5_reflexion['list_thinking'][0].content}; answer: {results5_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results5_reflexion['critic_agent'].id}, feedback round {i}, thinking: {results5_reflexion['list_feedback'][i].content}; answer: {results5_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results5_reflexion['cot_agent'].id}, refining final product matching round {i+1}, thinking: {results5_reflexion['list_thinking'][i+1].content}; answer: {results5_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 5 Reflexion output: thinking - {results5_reflexion['thinking'].content}; answer - {results5_reflexion['answer'].content}")
    logs.append(results5_reflexion['subtask_desc'])

    final_answer = await self.make_final_answer(results5_reflexion['thinking'], results5_reflexion['answer'], sub_tasks, agents)
    return final_answer, logs
