async def forward_197(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Extract and list all given data and parameters from the query: total cobalt concentration, SCN- concentration, and stability constants (β1, β2, β3, β4) for cobalt(II) thiocyanato complexes. Confirm extracted values match the original question to avoid transcription errors."
    cot_agent_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_agent_desc_1,
        n_repeat=self.max_sc
    )
    agents.append(f"CoT-SC agent {results1['cot_agent'][0].id}, extracting data and parameters, thinking: {results1['list_thinking'][0]}; answer: {results1['list_answer'][0]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['list_thinking'][0]}; answer - {results1['list_answer'][0]}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = "Subtask 2: Identify and write down all cobalt(II) species formed in solution with thiocyanate, including free Co(II) and complexes Co(SCN)+, Co(SCN)2, Co(SCN)3, Co(SCN)4, and express their formation equilibria using the given stability constants with correct stoichiometry."
    cot_agent_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo, results1['list_thinking'][0], results1['list_answer'][0]],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_agent_desc_2,
        n_repeat=self.max_sc
    )
    agents.append(f"CoT-SC agent {results2['cot_agent'][0].id}, identifying species and equilibria, thinking: {results2['list_thinking'][0]}; answer: {results2['list_answer'][0]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['list_thinking'][0]}; answer - {results2['list_answer'][0]}")
    logs.append(results2['subtask_desc'])

    cot_instruction_3 = "Subtask 3: Formulate the mass balance equation for total cobalt concentration as the sum of free Co(II) and all complex species concentrations, explicitly stating that all species concentrations sum to total cobalt concentration."
    cot_agent_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo, results2['list_thinking'][0], results2['list_answer'][0]],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_agent_desc_3,
        n_repeat=self.max_sc
    )
    agents.append(f"CoT-SC agent {results3['cot_agent'][0].id}, formulating Co mass balance, thinking: {results3['list_thinking'][0]}; answer: {results3['list_answer'][0]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['list_thinking'][0]}; answer - {results3['list_answer'][0]}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = "Subtask 4: Formulate the mass balance equation for total thiocyanate concentration, considering free SCN- and SCN- bound in complexes, and reach a final consensus confirming the equation comprehensively accounts for stoichiometry."
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'context': ["user query", results2['list_thinking'][0], results2['list_answer'][0]],
        'input': [taskInfo, results2['list_thinking'][0], results2['list_answer'][0]],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    final_decision_desc_4 = {
        'instruction': "Subtask 4: Make final decision on the thiocyanate mass balance equation.",
        'output': ['thinking', 'answer'],
        'temperature': 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc=final_decision_desc_4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, formulating SCN- mass balance, thinking: {results4['list_thinking'][round][idx]}; answer: {results4['list_answer'][round][idx]}")
    agents.append(f"Final Decision agent, thiocyanate mass balance, thinking: {results4['thinking']}; answer: {results4['answer']}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking']}; answer - {results4['answer']}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction_5 = "Subtask 5: Using the equilibrium expressions and mass balances, explicitly derive the denominator D = 1 + β1[SCN] + β2[SCN]^2 + β3[SCN]^3 + β4[SCN]^4, calculate each species fraction f_n = β_n[SCN]^n / D, show intermediate steps, verify that the sum of all species fractions equals 1, and calculate the concentrations of all cobalt species (free Co(II), Co(SCN)+, Co(SCN)2, Co(SCN)3, Co(SCN)4)."
    critic_instruction_5 = "Please review the calculated species fractions and concentrations, verify numeric consistency, and provide feedback on their validity."
    cot_reflect_desc_5 = {
        'instruction': cot_reflect_instruction_5,
        'input': [taskInfo, results3['list_thinking'][0], results3['list_answer'][0], results4['thinking'], results4['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc_5 = {
        'instruction': critic_instruction_5,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc_5,
        critic_desc=critic_desc_5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, calculating species fractions and concentrations, thinking: {results5['list_thinking'][0]}; answer: {results5['list_answer'][0]}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback round {i}, thinking: {results5['list_feedback'][i]}; answer: {results5['list_correct'][i]}")
        agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining calculations round {i}, thinking: {results5['list_thinking'][i+1]}; answer: {results5['list_answer'][i+1]}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking']}; answer - {results5['answer']}")
    logs.append(results5['subtask_desc'])

    cot_instruction_6 = "Subtask 6: Calculate the percentage of the blue dithiocyanato cobalt(II) complex (Co(SCN)2) relative to the total cobalt concentration using the concentrations obtained in Subtask 5. Explicitly apply the formula: percentage = ([Co(SCN)2] / total Co concentration) × 100, and cross-check the calculation for consistency."
    cot_agent_desc_6 = {
        'instruction': cot_instruction_6,
        'input': [taskInfo, results5['list_thinking'][0], results5['list_answer'][0]],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6 = await self.sc_cot(
        subtask_id="subtask_6",
        cot_sc_desc=cot_agent_desc_6,
        n_repeat=self.max_sc
    )
    agents.append(f"CoT-SC agent {results6['cot_agent'][0].id}, calculating percentage of Co(SCN)2, thinking: {results6['list_thinking'][0]}; answer: {results6['list_answer'][0]}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['list_thinking'][0]}; answer - {results6['list_answer'][0]}")
    logs.append(results6['subtask_desc'])

    debate_instruction_7 = "Subtask 7: Compare the calculated percentage of Co(SCN)2 with the given multiple-choice options (16.9%, 42.3%, 25.6%, 38.1%) and select the correct answer choice. Defend the selection by showing which option matches the computed value and why others do not."
    debate_desc_7 = {
        'instruction': debate_instruction_7,
        'context': ["user query", results6['list_thinking'][0], results6['list_answer'][0]],
        'input': [taskInfo, results6['list_thinking'][0], results6['list_answer'][0]],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    final_decision_desc_7 = {
        'instruction': "Subtask 7: Make final decision on the correct multiple-choice answer based on the comparison.",
        'output': ['thinking', 'answer'],
        'temperature': 0.0
    }
    results7 = await self.debate(
        subtask_id="subtask_7",
        debate_desc=debate_desc_7,
        final_decision_desc=final_decision_desc_7,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results7['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, selecting final answer, thinking: {results7['list_thinking'][round][idx]}; answer: {results7['list_answer'][round][idx]}")
    agents.append(f"Final Decision agent, final answer selection, thinking: {results7['thinking']}; answer: {results7['answer']}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking']}; answer - {results7['answer']}")
    logs.append(results7['subtask_desc'])

    cot_reflect_instruction_8 = "Subtask 8: Perform a final sanity check verifying that the sum of all species fractions equals 1 and that the percentage calculation is consistent. If discrepancies arise, revise calculations before finalizing the answer."
    critic_instruction_8 = "Please review the overall numeric consistency and confirm the final answer's validity."
    cot_reflect_desc_8 = {
        'instruction': cot_reflect_instruction_8,
        'input': [taskInfo, results5['list_thinking'][0], results5['list_answer'][0], results6['list_thinking'][0], results6['list_answer'][0], results7['thinking'], results7['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6", "answer of subtask 6", "thinking of subtask 7", "answer of subtask 7"]
    }
    critic_desc_8 = {
        'instruction': critic_instruction_8,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results8 = await self.reflexion(
        subtask_id="subtask_8",
        cot_reflect_desc=cot_reflect_desc_8,
        critic_desc=critic_desc_8,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results8['cot_agent'].id}, performing final sanity check, thinking: {results8['list_thinking'][0]}; answer: {results8['list_answer'][0]}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results8['critic_agent'].id}, feedback round {i}, thinking: {results8['list_feedback'][i]}; answer: {results8['list_correct'][i]}")
        agents.append(f"Reflexion CoT agent {results8['cot_agent'].id}, refining sanity check round {i}, thinking: {results8['list_thinking'][i+1]}; answer: {results8['list_answer'][i+1]}")
    sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking']}; answer - {results8['answer']}")
    logs.append(results8['subtask_desc'])

    final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
    return final_answer, logs
