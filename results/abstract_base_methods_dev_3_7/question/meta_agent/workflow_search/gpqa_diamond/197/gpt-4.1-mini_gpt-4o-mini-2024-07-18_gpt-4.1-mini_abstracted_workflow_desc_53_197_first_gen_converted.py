async def forward_197(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Calculate the concentrations of all cobalt(II) thiocyanato complexes (Co(SCN)+, Co(SCN)2, Co(SCN)3-, Co(SCN)4^2-) using the given total cobalt concentration, SCN- concentration, and stability constants β1=9, β2=40, β3=63, β4=16."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results1['list_thinking']):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, calculating cobalt(II) thiocyanato complexes concentrations, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction2 = "Sub-task 2: Calculate the concentration of free (uncomplexed) Co(II) ions by subtracting the sum of all complexed cobalt species concentrations from the total cobalt concentration, based on Sub-task 1 outputs."
    critic_instruction2 = "Please review the calculation of free Co(II) concentration and provide any limitations or corrections needed."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, calculating free Co(II) concentration, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results2['critic_agent'].id}, providing feedback, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining free Co(II) concentration, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = "Sub-task 3: Sum the concentrations of all cobalt-containing species (free Co(II) and all complexes) to verify total cobalt balance and prepare for fraction calculation, based on outputs from Sub-tasks 1 and 2."
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_instruction=cot_instruction3,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, summing cobalt species concentrations, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")

    cot_reflect_instruction3 = "Sub-task 3: Based on the sum of cobalt species concentrations, verify the total cobalt balance and identify any discrepancies or errors."
    critic_instruction3 = "Please review the total cobalt balance verification and provide feedback on accuracy and limitations."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
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
    agents.append(f"Reflexion CoT agent {results3_reflexion['cot_agent'].id}, verifying total cobalt balance, thinking: {results3_reflexion['list_thinking'][0].content}; answer: {results3_reflexion['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results3_reflexion['critic_agent'].id}, providing feedback, thinking: {results3_reflexion['list_feedback'][i].content}; answer: {results3_reflexion['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results3_reflexion['cot_agent'].id}, refining total cobalt balance verification, thinking: {results3_reflexion['list_thinking'][i + 1].content}; answer: {results3_reflexion['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3_reflexion['thinking'].content}; answer - {results3_reflexion['answer'].content}")
    logs.append(results3_reflexion['subtask_desc'])

    debate_instruction4 = "Sub-task 4: Calculate the percentage of the blue dithiocyanato cobalt(II) complex (Co(SCN)2) among all cobalt species by dividing its concentration by the total cobalt concentration and multiplying by 100%, based on the verified total cobalt balance."
    final_decision_instruction4 = "Sub-task 4: Make final decision on the percentage of the blue dithiocyanato cobalt(II) complex among all cobalt species."
    debate_desc4 = {
        "instruction": debate_instruction4,
        "context": ["user query", results3_reflexion['thinking'], results3_reflexion['answer']],
        "input": [taskInfo, results3_reflexion['thinking'], results3_reflexion['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc4 = {
        "instruction": final_decision_instruction4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        final_decision_desc=final_decision_desc4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, calculating percentage of blue dithiocyanato complex, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating final percentage, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
