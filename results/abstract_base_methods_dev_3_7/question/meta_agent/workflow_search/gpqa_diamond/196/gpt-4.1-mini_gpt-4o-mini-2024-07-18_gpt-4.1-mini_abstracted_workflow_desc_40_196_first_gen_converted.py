async def forward_196(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Extract and characterize key functional groups and structural features from the IR spectrum of Compound X, based on the given IR data and task context."
    cot_agent_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc_1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing IR spectrum, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction_2 = "Subtask 2: Extract and characterize key proton environments and structural features from the 1H NMR spectrum of Compound X, considering the given NMR data and task context."
    cot_sc_desc_2 = {
        'instruction': cot_sc_instruction_2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc_2,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, analyzing NMR spectrum, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction_3 = "Subtask 3: Integrate IR and NMR data to deduce the likely structure or functional groups present in Compound X before reaction, using the outputs from Subtask 1 and Subtask 2."
    cot_sc_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc_3,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, integrating IR and NMR data, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = "Subtask 4: Analyze the reaction conditions (red phosphorus and HI) to determine the type of chemical transformation Compound X will undergo, considering the chemical context and outputs from previous subtasks."
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'context': ["user query"],
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc={
            'instruction': "Subtask 4: Make final decision on the type of chemical transformation Compound X undergoes.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, analyzing reaction conditions, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding reaction type, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction_5 = "Subtask 5: Predict the final product structure of Compound X after reaction with red phosphorus and HI based on the deduced initial structure and reaction type from Subtask 3 and Subtask 4."
    final_decision_instruction_5 = "Subtask 5: Make final decision on the predicted final product structure."
    debate_desc_5 = {
        'instruction': debate_instruction_5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc_5 = {
        'instruction': final_decision_instruction_5,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc_5,
        final_decision_desc=final_decision_desc_5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, predicting final product, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding final product, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    debate_instruction_6 = "Subtask 6: Compare the predicted final product structure with the given multiple-choice options and select the correct answer."
    final_decision_instruction_6 = "Subtask 6: Make final decision on the correct multiple-choice answer."
    debate_desc_6 = {
        'instruction': debate_instruction_6,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc_6 = {
        'instruction': final_decision_instruction_6,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc_6,
        final_decision_desc=final_decision_desc_6,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results6['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, selecting correct answer, thinking: {results6['list_thinking'][round][idx].content}; answer: {results6['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, selecting final answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
