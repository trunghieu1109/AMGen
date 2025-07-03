async def forward_197(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Identify all cobalt(II) thiocyanato complex species formed in solution based on the given stability constants β1=9, β2=40, β3=63, and β4=16, and write their formation equilibria."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction1 = "Subtask 1 Reflexion: Review the identified cobalt(II) thiocyanato complex species and their formation equilibria for correctness and completeness."
    critic_instruction1 = "Please review the identified species and equilibria and provide feedback on any missing or incorrect information."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo] + results1['list_thinking'] + results1['list_answer'],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1_reflexion = await self.reflexion(
        subtask_id="subtask_1_reflexion",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results1['cot_agent']]}, identifying cobalt(II) thiocyanato complexes, thinking: {results1['list_thinking'][0]}; answer: {results1['list_answer'][0]}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results1_reflexion['cot_agent'].id}, round {i}, reviewing species identification, thinking: {results1_reflexion['list_thinking'][i]}; answer: {results1_reflexion['list_answer'][i]}")
        agents.append(f"Critic agent {results1_reflexion['critic_agent'].id}, round {i}, feedback: {results1_reflexion['list_feedback'][i]}; correctness: {results1_reflexion['list_correct'][i]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1_reflexion['thinking'].content}; answer - {results1_reflexion['answer'].content}")
    logs.append(results1_reflexion['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Express the total cobalt concentration as the sum of free Co(II) and all complexed species concentrations, and write the mass balance and equilibrium expressions using the given β values and initial concentrations c(Co) = 10^-2 M and [SCN-] = 0.1 M, based on Subtask 1 output."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction2,
        input_list=[taskInfo, results1_reflexion['thinking'], results1_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction2 = "Subtask 2 Reflexion: Review the mass balance and equilibrium expressions for correctness and consistency with the given data and Subtask 1 output."
    critic_instruction2 = "Please review the mass balance and equilibrium expressions and provide feedback on any errors or omissions."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1_reflexion['thinking'], results1_reflexion['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2_reflexion = await self.reflexion(
        subtask_id="subtask_2_reflexion",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results2['cot_agent']]}, formulating mass balance and equilibrium expressions, thinking: {results2['list_thinking'][0]}; answer: {results2['list_answer'][0]}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results2_reflexion['cot_agent'].id}, round {i}, reviewing mass balance, thinking: {results2_reflexion['list_thinking'][i]}; answer: {results2_reflexion['list_answer'][i]}")
        agents.append(f"Critic agent {results2_reflexion['critic_agent'].id}, round {i}, feedback: {results2_reflexion['list_feedback'][i]}; correctness: {results2_reflexion['list_correct'][i]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2_reflexion['thinking'].content}; answer - {results2_reflexion['answer'].content}")
    logs.append(results2_reflexion['subtask_desc'])

    cot_sc_instruction3 = "Subtask 3: Calculate the equilibrium concentrations of all cobalt species (free Co(II), mono-, di-, tri-, and tetrathiocyanato complexes) using the mass balance and stability constants, considering the initial concentrations c(Co) = 10^-2 M and [SCN-] = 0.1 M, based on Subtask 2 output."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction3,
        input_list=[taskInfo, results2_reflexion['thinking'], results2_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction3 = "Subtask 3 Reflexion: Review the calculated equilibrium concentrations for accuracy and consistency with the given data and previous subtasks."
    critic_instruction3 = "Please review the equilibrium concentrations and provide feedback on any calculation errors or inconsistencies."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results2_reflexion['thinking'], results2_reflexion['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc3 = {
        'instruction': critic_instruction3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3_reflexion = await self.reflexion(
        subtask_id="subtask_3_reflexion",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results3['cot_agent']]}, calculating equilibrium concentrations, thinking: {results3['list_thinking'][0]}; answer: {results3['list_answer'][0]}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results3_reflexion['cot_agent'].id}, round {i}, reviewing concentrations, thinking: {results3_reflexion['list_thinking'][i]}; answer: {results3_reflexion['list_answer'][i]}")
        agents.append(f"Critic agent {results3_reflexion['critic_agent'].id}, round {i}, feedback: {results3_reflexion['list_feedback'][i]}; correctness: {results3_reflexion['list_correct'][i]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3_reflexion['thinking'].content}; answer - {results3_reflexion['answer'].content}")
    logs.append(results3_reflexion['subtask_desc'])

    cot_sc_instruction4 = "Subtask 4: Determine the fraction of the blue dithiocyanato cobalt(II) complex (species with two SCN- ligands) relative to the total cobalt concentration by dividing its equilibrium concentration by the total cobalt concentration, based on Subtask 3 output."
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_instruction=cot_sc_instruction4,
        input_list=[taskInfo, results3_reflexion['thinking'], results3_reflexion['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 3", "answer of subtask 3"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction4 = "Subtask 4 Reflexion: Review the calculated fraction of the dithiocyanato complex for correctness and consistency."
    critic_instruction4 = "Please review the fraction calculation and provide feedback on any errors or inconsistencies."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3_reflexion['thinking'], results3_reflexion['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4_reflexion = await self.reflexion(
        subtask_id="subtask_4_reflexion",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in results4['cot_agent']]}, determining fraction of dithiocyanato complex, thinking: {results4['list_thinking'][0]}; answer: {results4['list_answer'][0]}")
    for i in range(self.max_round):
        agents.append(f"Reflexion CoT agent {results4_reflexion['cot_agent'].id}, round {i}, reviewing fraction, thinking: {results4_reflexion['list_thinking'][i]}; answer: {results4_reflexion['list_answer'][i]}")
        agents.append(f"Critic agent {results4_reflexion['critic_agent'].id}, round {i}, feedback: {results4_reflexion['list_feedback'][i]}; correctness: {results4_reflexion['list_correct'][i]}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4_reflexion['thinking'].content}; answer - {results4_reflexion['answer'].content}")
    logs.append(results4_reflexion['subtask_desc'])

    debate_instruction5 = "Subtask 5: Convert the fraction of the dithiocyanato complex to a percentage and compare it with the given multiple-choice options (16.9%, 42.3%, 25.6%, 38.1%) to select the correct answer."
    final_decision_instruction5 = "Subtask 5: Make final decision on the correct percentage choice of the blue dithiocyanato cobalt(II) complex among all cobalt species."
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
            agents.append(f"Debate agent {agent.id}, round {round}, converting fraction to percentage and selecting answer, thinking: {results5['list_thinking'][round][idx]}; answer: {results5['list_answer'][round][idx]}")
    agents.append(f"Final Decision agent, calculating final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
