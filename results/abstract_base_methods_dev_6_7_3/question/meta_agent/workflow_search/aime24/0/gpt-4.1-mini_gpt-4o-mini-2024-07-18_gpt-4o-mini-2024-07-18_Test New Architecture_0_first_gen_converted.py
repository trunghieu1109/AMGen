async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Define variables s for walking speed (km/h) and t for coffee-shop time (minutes) based on the problem context."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, defining variables s and t, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Formulate equation for total time when walking at speed s: 9/s + t/60 = 4 hours."
    cot_agent_desc = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, formulating first equation, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = "Subtask 3: Formulate equation for total time when walking at speed s+2: 9/(s+2) + t/60 = 2 hours 24 minutes."
    cot_agent_desc = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, formulating second equation, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_instruction4 = "Subtask 4: Solve the two equations simultaneously to find numeric values of s and t."
    cot_agent_desc = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, solving equations for s and t, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction6 = "Subtask 6: Combine the numeric results for s and t to prepare inputs for further computations."
    critic_instruction6 = "Please review the combined numeric results for s and t and provide any limitations or corrections needed."
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc6 = {
        'instruction': critic_instruction6,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc6,
        critic_desc=critic_desc6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, combining numeric results for s and t, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, providing feedback, thinking: {results6['list_feedback'][i].content}; answer: {results6['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_sc_instruction5 = "Subtask 5: Compare walking times at speed s and at speed s+0.5 to quantify the change."
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    for idx, key in enumerate(results5['list_thinking']):
        agents.append(f"CoT-SC agent {results5['cot_agent'][idx].id}, comparing walking times at s and s+0.5, thinking: {results5['list_thinking'][idx]}; answer: {results5['list_answer'][idx]}")
    logs.append(results5['subtask_desc'])

    cot_sc_instruction4 = "Subtask 4 (reused id avoided): Calculate walking time at speed s+0.5 km/h, add t/60, and convert to minutes."
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7 = await self.sc_cot(
        subtask_id="subtask_7",
        cot_sc_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    for idx, key in enumerate(results7['list_thinking']):
        agents.append(f"CoT-SC agent {results7['cot_agent'][idx].id}, calculating walking time at s+0.5, thinking: {results7['list_thinking'][idx]}; answer: {results7['list_answer'][idx]}")
    logs.append(results7['subtask_desc'])

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
