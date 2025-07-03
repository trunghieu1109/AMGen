async def forward_186(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Understand and extract the detection criteria and ESPRESSO spectrograph capabilities relevant to S/N ratio, exposure time, and telescope parameters from the provided instrument overview."
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user input"
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing ESPRESSO capabilities, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction_1 = "Subtask 1 Reflexion: Review and refine the understanding of detection criteria and ESPRESSO spectrograph capabilities to ensure accuracy and completeness."
    critic_instruction_1 = "Please review the extracted detection criteria and ESPRESSO capabilities and provide feedback on any missing or incorrect information."
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
    agents.append(f"Reflexion CoT agent {results1_reflexion['cot_agent'].id}, refining ESPRESSO capabilities, thinking: {results1_reflexion['list_thinking'][0].content}; answer: {results1_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results1_reflexion['critic_agent'].id}, feedback: {results1_reflexion['list_feedback'][i].content}; correction: {results1_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results1_reflexion['cot_agent'].id}, refining ESPRESSO capabilities, thinking: {results1_reflexion['list_thinking'][i+1].content}; answer: {results1_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 1 Reflexion output: thinking - {results1_reflexion['thinking'].content}; answer - {results1_reflexion['answer'].content}")
    logs.append(results1_reflexion['subtask_desc'])

    cot_sc_instruction_2 = "Subtask 2: Calculate the apparent V magnitude for each star using the given absolute V magnitude and distance, applying the distance modulus formula, based on the understanding from Subtask 1."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction_2,
        input_list=[taskInfo, results1_reflexion['thinking'], results1_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1 reflexion", "answer of subtask 1 reflexion"],
        n_repeat=self.max_sc
    )
    for idx, thinking in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, calculating apparent magnitudes, thinking: {thinking}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    debate_instruction_2 = "Subtask 2 Debate: Discuss and verify the calculated apparent magnitudes for accuracy and consistency."
    final_decision_instruction_2 = "Subtask 2 Debate: Make final decision on the apparent magnitudes calculated."
    debate_desc_2 = {
        "instruction": debate_instruction_2,
        "context": ["user query", results1_reflexion['thinking'].content, results1_reflexion['answer'].content],
        "input": [taskInfo, results1_reflexion['thinking'], results1_reflexion['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_2 = {
        "instruction": final_decision_instruction_2,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results2_debate = await self.debate(
        subtask_id="subtask_2_debate",
        debate_desc=debate_desc_2,
        final_decision_desc=final_decision_desc_2,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results2_debate['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, verifying apparent magnitudes, thinking: {results2_debate['list_thinking'][round][idx].content}; answer: {results2_debate['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding apparent magnitudes, thinking: {results2_debate['thinking'].content}; answer: {results2_debate['answer'].content}")
    sub_tasks.append(f"Subtask 2 Debate output: thinking - {results2_debate['thinking'].content}; answer - {results2_debate['answer'].content}")
    logs.append(results2_debate['subtask_desc'])

    cot_sc_instruction_3 = "Subtask 3: Estimate the expected signal-to-noise ratio (S/N) for each star during a 1-hour exposure with ESPRESSO on an 8m VLT telescope, using the apparent magnitudes and instrument sensitivity parameters from previous subtasks."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction_3,
        input_list=[taskInfo, results2_debate['thinking'], results2_debate['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results2_debate['thinking'].content, results2_debate['answer'].content],
        n_repeat=self.max_sc
    )
    agents.append(f"CoT-SC agent {results3['cot_agent'][0].id}, estimating S/N ratios, thinking: {results3['list_thinking'][0]}; answer: {results3['list_answer'][0]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_reflect_instruction_3 = "Subtask 3 Reflexion: Review and refine the S/N ratio estimations to ensure they meet the detection criteria accurately."
    critic_instruction_3 = "Please review the S/N ratio estimations and provide feedback on their validity and any limitations."
    cot_reflect_desc_3 = {
        'instruction': cot_reflect_instruction_3,
        'input': [taskInfo, results2_debate['thinking'], results2_debate['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", results2_debate['thinking'].content, results2_debate['answer'].content, results3['thinking'].content, results3['answer'].content]
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
    agents.append(f"Reflexion CoT agent {results3_reflexion['cot_agent'].id}, refining S/N estimations, thinking: {results3_reflexion['list_thinking'][0].content}; answer: {results3_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results3_reflexion['critic_agent'].id}, feedback: {results3_reflexion['list_feedback'][i].content}; correction: {results3_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results3_reflexion['cot_agent'].id}, refining S/N estimations, thinking: {results3_reflexion['list_thinking'][i+1].content}; answer: {results3_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 3 Reflexion output: thinking - {results3_reflexion['thinking'].content}; answer - {results3_reflexion['answer'].content}")
    logs.append(results3_reflexion['subtask_desc'])

    debate_instruction_4 = "Subtask 4: Determine which stars meet or exceed the detection threshold of S/N ≥ 10 per binned pixel in 1 hour, marking them as detectable based on S/N estimations."
    final_decision_instruction_4 = "Subtask 4: Make final decision on detectability of stars based on S/N threshold."
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", results3_reflexion['thinking'].content, results3_reflexion['answer'].content],
        "input": [taskInfo, results3_reflexion['thinking'], results3_reflexion['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_4 = {
        "instruction": final_decision_instruction_4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc=final_decision_desc_4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, determining detectability, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding detectability, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_instruction_5 = "Subtask 5: Count the total number of detectable stars from the list and map this count to the provided multiple-choice options to select the correct answer."
    results5 = await self.cot(
        subtask_id="subtask_5",
        cot_instruction=cot_instruction_5,
        input_list=[taskInfo, results4['thinking'], results4['answer'], results1_reflexion['thinking'], results1_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", results4['thinking'].content, results4['answer'].content, results1_reflexion['thinking'].content, results1_reflexion['answer'].content]
    )
    reflexion_instruction_5 = "Subtask 5 Reflexion: Review the counting and mapping of detectable stars to multiple-choice answers for correctness and completeness."
    critic_instruction_5 = "Please review the counting and mapping results and provide feedback on any errors or improvements."
    cot_reflect_desc_5 = {
        'instruction': reflexion_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", results4['thinking'].content, results4['answer'].content, results5['thinking'].content, results5['answer'].content]
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
    agents.append(f"Reflexion CoT agent {results5_reflexion['cot_agent'].id}, refining final count and mapping, thinking: {results5_reflexion['list_thinking'][0].content}; answer: {results5_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results5_reflexion['critic_agent'].id}, feedback: {results5_reflexion['list_feedback'][i].content}; correction: {results5_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results5_reflexion['cot_agent'].id}, refining final count and mapping, thinking: {results5_reflexion['list_thinking'][i+1].content}; answer: {results5_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 5 Reflexion output: thinking - {results5_reflexion['thinking'].content}; answer - {results5_reflexion['answer'].content}")
    logs.append(results5_reflexion['subtask_desc'])

    final_answer = await self.make_final_answer(results5_reflexion['thinking'], results5_reflexion['answer'], sub_tasks, agents)
    return final_answer, logs
